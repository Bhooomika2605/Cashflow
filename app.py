from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import json
import sqlite3
import numpy as np
from sklearn.ensemble import IsolationForest
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Database
def init_db():
    conn = sqlite3.connect('kirana_store.db')
    c = conn.cursor()
    
    # Transactions table
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  item TEXT,
                  amount REAL,
                  type TEXT,
                  payment_method TEXT,
                  customer_name TEXT)''')
    
    # Inventory table
    c.execute('''CREATE TABLE IF NOT EXISTS inventory
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  item_name TEXT UNIQUE,
                  quantity INTEGER,
                  reorder_level INTEGER,
                  unit_price REAL,
                  last_updated TEXT)''')
    
    # Alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  alert_type TEXT,
                  message TEXT,
                  status TEXT)''')
    
    conn.commit()
    conn.close()

init_db()

# ===== NLP PARSER =====
def parse_transaction_text(text):
    """
    Extract transaction details from text using regex patterns
    """
    text = text.lower()
    
    # Extract amount
    amount_pattern = r'(?:rs\.?|rupees?|₹)\s*(\d+(?:\.\d{2})?)|(\d+(?:\.\d{2})?)\s*(?:rs\.?|rupees?|₹)'
    amount_match = re.search(amount_pattern, text)
    amount = float(amount_match.group(1) or amount_match.group(2)) if amount_match else 0.0
    
    # Detect transaction type
    transaction_type = 'sale'
    if any(word in text for word in ['buy', 'purchase', 'bought', 'supplier']):
        transaction_type = 'purchase'
    
    # Extract item (simplified - look for common items)
    items = ['rice', 'wheat', 'sugar', 'oil', 'dal', 'tea', 'salt', 'milk', 'biscuit']
    item = 'general'
    for i in items:
        if i in text:
            item = i
            break
    
    # Extract quantity
    quantity_pattern = r'(\d+)\s*(?:kg|kg\.|kilos?|packets?|units?|pieces?)'
    quantity_match = re.search(quantity_pattern, text)
    quantity = int(quantity_match.group(1)) if quantity_match else 1
    
    return {
        'item': item,
        'amount': amount,
        'type': transaction_type,
        'quantity': quantity,
        'payment_method': 'cash'  # Default
    }

# ===== AGENT 1: CASH FLOW FORECAST =====
def cash_flow_forecast():
    """
    Predict if cash shortage is likely in next 7 days
    """
    conn = sqlite3.connect('kirana_store.db')
    c = conn.cursor()
    
    # Get last 30 days transactions
    c.execute('''SELECT amount, type FROM transactions 
                 WHERE timestamp >= date('now', '-30 days')''')
    transactions = c.fetchall()
    conn.close()
    
    if not transactions:
        return {'shortage_predicted': False, 'recommendation': 'Insufficient data'}
    
    # Calculate daily cash flow
    sales = sum([t[0] for t in transactions if t[1] == 'sale'])
    purchases = sum([t[0] for t in transactions if t[1] == 'purchase'])
    net_flow = sales - purchases
    daily_avg = net_flow / 30
    
    # Simple prediction: if average is negative, shortage predicted
    shortage_predicted = daily_avg < 0
    
    recommendation = 'No action needed'
    if shortage_predicted:
        recommendation = 'Cash shortage expected. Consider delaying purchases or securing short-term loan.'
    
    return {
        'shortage_predicted': shortage_predicted,
        'daily_avg_flow': round(daily_avg, 2),
        'recommendation': recommendation
    }

# ===== AGENT 2: INVENTORY ALERT =====
def inventory_alert():
    """
    Check for low stock items
    """
    conn = sqlite3.connect('kirana_store.db')
    c = conn.cursor()
    
    c.execute('''SELECT item_name, quantity, reorder_level 
                 FROM inventory 
                 WHERE quantity <= reorder_level''')
    low_stock_items = c.fetchall()
    conn.close()
    
    stock_low = len(low_stock_items) > 0
    
    items_to_reorder = [{'item': item[0], 'current': item[1], 'reorder_level': item[2]} 
                        for item in low_stock_items]
    
    return {
        'stock_low': stock_low,
        'items_to_reorder': items_to_reorder
    }

# ===== AGENT 3: FRAUD DETECTION =====
def fraud_detection(transaction_amount):
    """
    Detect if current transaction is anomalous
    """
    conn = sqlite3.connect('kirana_store.db')
    c = conn.cursor()
    
    c.execute('SELECT amount FROM transactions WHERE type = "sale"')
    amounts = [row[0] for row in c.fetchall()]
    conn.close()
    
    if len(amounts) < 10:
        return {'fraud_suspected': False, 'reason': 'Insufficient data'}
    
    # Statistical anomaly detection
    mean_amount = np.mean(amounts)
    std_amount = np.std(amounts)
    
    # Flag if transaction is > 3 standard deviations
    threshold = mean_amount + (3 * std_amount)
    fraud_suspected = transaction_amount > threshold
    
    reason = f'Transaction amount (₹{transaction_amount}) exceeds normal range (₹{round(threshold, 2)})'
    
    return {
        'fraud_suspected': fraud_suspected,
        'reason': reason if fraud_suspected else 'Transaction within normal range'
    }

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process-input', methods=['POST'])
def process_input():
    """
    Main endpoint to process voice/text input
    """
    try:
        data = request.json
        input_text = data.get('text', '')
        
        if not input_text:
            return jsonify({'error': 'No input provided'}), 400
        
        # Parse transaction
        parsed_data = parse_transaction_text(input_text)
        
        # Store in database
        conn = sqlite3.connect('kirana_store.db')
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        c.execute('''INSERT INTO transactions 
                     (timestamp, item, amount, type, payment_method, customer_name)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (timestamp, parsed_data['item'], parsed_data['amount'], 
                   parsed_data['type'], parsed_data['payment_method'], 'Customer'))
        
        # Update inventory
        if parsed_data['type'] == 'sale':
            c.execute('''UPDATE inventory SET quantity = quantity - ? 
                         WHERE item_name = ?''',
                      (parsed_data['quantity'], parsed_data['item']))
        else:
            c.execute('''INSERT OR REPLACE INTO inventory 
                         (item_name, quantity, reorder_level, unit_price, last_updated)
                         VALUES (?, 
                                 COALESCE((SELECT quantity FROM inventory WHERE item_name = ?), 0) + ?,
                                 10, ?, ?)''',
                      (parsed_data['item'], parsed_data['item'], parsed_data['quantity'],
                       parsed_data['amount'] / parsed_data['quantity'], timestamp))
        
        conn.commit()
        conn.close()
        
        # Run AI agents
        cf_result = cash_flow_forecast()
        inv_result = inventory_alert()
        fraud_result = fraud_detection(parsed_data['amount'])
        
        # Generate alerts
        alerts = []
        
        if cf_result['shortage_predicted']:
            alerts.append({
                'type': 'cash_flow',
                'message': cf_result['recommendation'],
                'severity': 'high'
            })
        
        if inv_result['stock_low']:
            for item in inv_result['items_to_reorder']:
                alerts.append({
                    'type': 'inventory',
                    'message': f"Low stock: {item['item']} - Reorder needed",
                    'severity': 'medium'
                })
        
        if fraud_result['fraud_suspected']:
            alerts.append({
                'type': 'fraud',
                'message': fraud_result['reason'],
                'severity': 'critical'
            })
        
        return jsonify({
            'success': True,
            'parsed_transaction': parsed_data,
            'agents': {
                'cash_flow': cf_result,
                'inventory': inv_result,
                'fraud': fraud_result
            },
            'alerts': alerts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard')
def dashboard():
    """
    Get dashboard data
    """
    conn = sqlite3.connect('kirana_store.db')
    c = conn.cursor()
    
    # Recent transactions
    c.execute('''SELECT * FROM transactions 
                 ORDER BY timestamp DESC LIMIT 10''')
    transactions = [dict(zip([col[0] for col in c.description], row)) 
                   for row in c.fetchall()]
    
    # Inventory status
    c.execute('SELECT * FROM inventory')
    inventory = [dict(zip([col[0] for col in c.description], row)) 
                for row in c.fetchall()]
    
    # Summary stats
    c.execute('''SELECT 
                    SUM(CASE WHEN type='sale' THEN amount ELSE 0 END) as total_sales,
                    SUM(CASE WHEN type='purchase' THEN amount ELSE 0 END) as total_purchases,
                    COUNT(*) as transaction_count
                 FROM transactions
                 WHERE date(timestamp) = date('now')''')
    stats = dict(zip([col[0] for col in c.description], c.fetchone()))
    
    conn.close()
    
    return jsonify({
        'transactions': transactions,
        'inventory': inventory,
        'stats': stats
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)