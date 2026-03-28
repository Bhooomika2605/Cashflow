from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime, timezone
import json
import sqlite3
import numpy as np
from sklearn.ensemble import IsolationForest
import re
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_PATH = 'kirana_store.db'

# Initialize Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
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

    # Reminders table
    c.execute('''CREATE TABLE IF NOT EXISTS reminders
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  message TEXT NOT NULL,
                  reminder_type TEXT NOT NULL,
                  frequency TEXT,
                  trigger_datetime TEXT,
                  trigger_time TEXT,
                  trigger_day INTEGER,
                  threshold_type TEXT,
                  threshold_value REAL,
                  timezone TEXT DEFAULT 'UTC',
                  is_active INTEGER DEFAULT 1,
                  created_at TEXT,
                  last_triggered TEXT,
                  notification_channels TEXT DEFAULT '["in_app"]')''')

    # Notifications table (in-app notification inbox)
    c.execute('''CREATE TABLE IF NOT EXISTS notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  reminder_id INTEGER,
                  title TEXT,
                  message TEXT,
                  status TEXT DEFAULT 'unread',
                  created_at TEXT,
                  FOREIGN KEY (reminder_id) REFERENCES reminders(id))''')
    
    conn.commit()
    conn.close()

init_db()

# ===== DATABASE HELPER =====
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
        conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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


# ===== REMINDER HELPERS =====

def _now_in_tz(timezone_str):
    """Return current datetime in the given timezone."""
    try:
        tz = pytz.timezone(timezone_str)
    except Exception:
        tz = pytz.utc
    return datetime.now(tz)


def _dispatch_notification(reminder_id, title, message, channels):
    """
    Dispatch a notification through the requested channels.
    - in_app: always persisted to the notifications table.
    - email / sms: scaffolded – extend the stubs below to wire up a real provider.
    """
    now_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    # 1. In-app notification (always stored)
    conn = get_db()
    c = conn.cursor()
    c.execute(
        '''INSERT INTO notifications (reminder_id, title, message, status, created_at)
           VALUES (?, ?, ?, 'unread', ?)''',
        (reminder_id, title, message, now_iso)
    )
    conn.commit()
    conn.close()

    # 2. Email (scaffolded – replace the stub with your SMTP / SendGrid call)
    if 'email' in channels:
        _send_email_stub(title, message)

    # 3. SMS (scaffolded – replace the stub with your Twilio / SMS provider call)
    if 'sms' in channels:
        _send_sms_stub(title, message)


def _send_email_stub(title, message):
    """Scaffold for email delivery. Wire up SMTP or an email API here."""
    # Example with smtplib (fill in credentials via environment variables):
    # import smtplib, os
    # from email.mime.text import MIMEText
    # msg = MIMEText(message)
    # msg['Subject'] = title
    # msg['From'] = os.environ.get('EMAIL_FROM', '')
    # msg['To'] = os.environ.get('EMAIL_TO', '')
    # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
    #     server.login(msg['From'], os.environ.get('EMAIL_PASSWORD', ''))
    #     server.send_message(msg)
    pass


def _send_sms_stub(title, message):
    """Scaffold for SMS delivery. Wire up Twilio or another SMS provider here."""
    # Example with Twilio (install twilio package and fill env vars):
    # from twilio.rest import Client
    # client = Client(os.environ.get('TWILIO_SID'), os.environ.get('TWILIO_TOKEN'))
    # client.messages.create(
    #     body=f"{title}: {message}",
    #     from_=os.environ.get('TWILIO_FROM'),
    #     to=os.environ.get('TWILIO_TO')
    # )
    pass


def _should_trigger_recurring(reminder, now_local):
    """
    Return True if a recurring reminder should fire right now.
    Checks are done at minute granularity to match the scheduler's 1-minute poll.
    """
    frequency = reminder['frequency']
    trigger_time = reminder['trigger_time']  # HH:MM
    trigger_day = reminder['trigger_day']    # 0-6 for weekly, 1-31 for monthly

    if not trigger_time:
        return False

    try:
        t_hour, t_minute = map(int, trigger_time.split(':'))
    except ValueError:
        return False

    if now_local.hour != t_hour or now_local.minute != t_minute:
        return False

    if frequency == 'daily':
        return True
    if frequency == 'weekly' and trigger_day is not None:
        return now_local.weekday() == int(trigger_day)
    if frequency == 'monthly' and trigger_day is not None:
        return now_local.day == int(trigger_day)
    return False


def _evaluate_threshold(reminder):
    """Check if a threshold-based reminder condition is met."""
    threshold_type = reminder['threshold_type']
    threshold_value = reminder['threshold_value']

    if threshold_type == 'cash_flow':
        result = cash_flow_forecast()
        if result.get('daily_avg_flow') is not None and threshold_value is not None:
            return result['daily_avg_flow'] < threshold_value
    elif threshold_type == 'inventory':
        result = inventory_alert()
        return result.get('stock_low', False)
    return False


def process_due_reminders():
    """
    Background job: evaluate every active reminder and fire notifications
    for those that are due.  Runs once per minute via APScheduler.
    """
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM reminders WHERE is_active = 1')
    reminders = [dict(r) for r in c.fetchall()]
    conn.close()

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    for reminder in reminders:
        channels = json.loads(reminder.get('notification_channels') or '["in_app"]')
        triggered = False

        if reminder['reminder_type'] == 'one_time':
            if not reminder['trigger_datetime']:
                continue
            try:
                trigger_dt = datetime.fromisoformat(reminder['trigger_datetime'])
            except ValueError:
                continue
            # Fire if the trigger time has just passed (within the last minute)
            delta = (now_utc - trigger_dt).total_seconds()
            if 0 <= delta < 60:
                triggered = True

        elif reminder['reminder_type'] == 'recurring':
            now_local = _now_in_tz(reminder.get('timezone') or 'UTC')
            triggered = _should_trigger_recurring(reminder, now_local)

        elif reminder['reminder_type'] == 'threshold':
            triggered = _evaluate_threshold(reminder)

        if triggered:
            # Avoid duplicate triggers within the same minute
            if reminder['last_triggered']:
                try:
                    last = datetime.fromisoformat(reminder['last_triggered'])
                    if (now_utc - last).total_seconds() < 60:
                        continue
                except ValueError:
                    pass

            _dispatch_notification(
                reminder['id'],
                reminder['title'],
                reminder['message'],
                channels
            )

            # Update last_triggered and deactivate one-time reminders
            conn = get_db()
            c = conn.cursor()
            if reminder['reminder_type'] == 'one_time':
                c.execute(
                    'UPDATE reminders SET last_triggered = ?, is_active = 0 WHERE id = ?',
                    (now_utc.isoformat(), reminder['id'])
                )
            else:
                c.execute(
                    'UPDATE reminders SET last_triggered = ? WHERE id = ?',
                    (now_utc.isoformat(), reminder['id'])
                )
            conn.commit()
            conn.close()


# ===== START BACKGROUND SCHEDULER =====
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=process_due_reminders,
    trigger=IntervalTrigger(minutes=1),
    id='reminder_processor',
    name='Process due reminders',
    replace_existing=True
)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())


# ===== REMINDER CRUD ENDPOINTS =====

@app.route('/api/reminders', methods=['GET'])
def list_reminders():
    """Return all reminders (active and inactive)."""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM reminders ORDER BY created_at DESC')
    reminders = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({'reminders': reminders})


@app.route('/api/reminders', methods=['POST'])
def create_reminder():
    """Create a new reminder."""
    try:
        data = request.json or {}
        title = data.get('title', '').strip()
        message = data.get('message', '').strip()
        reminder_type = data.get('reminder_type', 'one_time')

        if not title or not message:
            return jsonify({'error': 'title and message are required'}), 400

        if reminder_type not in ('one_time', 'recurring', 'threshold'):
            return jsonify({'error': 'reminder_type must be one_time, recurring, or threshold'}), 400

        frequency = data.get('frequency')
        trigger_datetime = data.get('trigger_datetime')
        trigger_time = data.get('trigger_time')
        trigger_day = data.get('trigger_day')
        threshold_type = data.get('threshold_type')
        threshold_value = data.get('threshold_value')
        tz_name = data.get('timezone', 'UTC')
        channels = data.get('notification_channels', ['in_app'])
        now_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        # Validate timezone
        try:
            pytz.timezone(tz_name)
        except pytz.exceptions.UnknownTimeZoneError:
            return jsonify({'error': f'Unknown timezone: {tz_name}'}), 400

        conn = get_db()
        c = conn.cursor()
        c.execute(
            '''INSERT INTO reminders
               (title, message, reminder_type, frequency, trigger_datetime,
                trigger_time, trigger_day, threshold_type, threshold_value,
                timezone, is_active, created_at, notification_channels)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)''',
            (title, message, reminder_type, frequency, trigger_datetime,
             trigger_time, trigger_day, threshold_type, threshold_value,
             tz_name, now_iso, json.dumps(channels))
        )
        reminder_id = c.lastrowid
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'id': reminder_id}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reminders/<int:reminder_id>', methods=['GET'])
def get_reminder(reminder_id):
    """Fetch a single reminder by ID."""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM reminders WHERE id = ?', (reminder_id,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return jsonify({'error': 'Reminder not found'}), 404
    return jsonify(dict(row))


@app.route('/api/reminders/<int:reminder_id>', methods=['PUT'])
def update_reminder(reminder_id):
    """Update an existing reminder."""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM reminders WHERE id = ?', (reminder_id,))
        existing = c.fetchone()
        if existing is None:
            conn.close()
            return jsonify({'error': 'Reminder not found'}), 404

        data = request.json or {}
        fields = {
            'title': data.get('title', existing['title']),
            'message': data.get('message', existing['message']),
            'reminder_type': data.get('reminder_type', existing['reminder_type']),
            'frequency': data.get('frequency', existing['frequency']),
            'trigger_datetime': data.get('trigger_datetime', existing['trigger_datetime']),
            'trigger_time': data.get('trigger_time', existing['trigger_time']),
            'trigger_day': data.get('trigger_day', existing['trigger_day']),
            'threshold_type': data.get('threshold_type', existing['threshold_type']),
            'threshold_value': data.get('threshold_value', existing['threshold_value']),
            'timezone': data.get('timezone', existing['timezone']),
            'is_active': int(data['is_active']) if 'is_active' in data else existing['is_active'],
            'notification_channels': json.dumps(
                data.get('notification_channels',
                         json.loads(existing['notification_channels'] or '["in_app"]'))
            ),
        }

        # Validate timezone if changed
        try:
            pytz.timezone(fields['timezone'])
        except pytz.exceptions.UnknownTimeZoneError:
            conn.close()
            return jsonify({'error': f"Unknown timezone: {fields['timezone']}"}), 400

        c.execute(
            '''UPDATE reminders SET
               title = ?, message = ?, reminder_type = ?, frequency = ?,
               trigger_datetime = ?, trigger_time = ?, trigger_day = ?,
               threshold_type = ?, threshold_value = ?, timezone = ?,
               is_active = ?, notification_channels = ?
               WHERE id = ?''',
            (fields['title'], fields['message'], fields['reminder_type'],
             fields['frequency'], fields['trigger_datetime'], fields['trigger_time'],
             fields['trigger_day'], fields['threshold_type'], fields['threshold_value'],
             fields['timezone'], fields['is_active'], fields['notification_channels'],
             reminder_id)
        )
        conn.commit()
        conn.close()
        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reminders/<int:reminder_id>', methods=['DELETE'])
def delete_reminder(reminder_id):
    """Delete a reminder and its associated notifications."""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id FROM reminders WHERE id = ?', (reminder_id,))
    if c.fetchone() is None:
        conn.close()
        return jsonify({'error': 'Reminder not found'}), 404
    c.execute('DELETE FROM notifications WHERE reminder_id = ?', (reminder_id,))
    c.execute('DELETE FROM reminders WHERE id = ?', (reminder_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ===== NOTIFICATION ENDPOINTS =====

@app.route('/api/notifications', methods=['GET'])
def list_notifications():
    """Return notifications, newest first. Optionally filter by status."""
    status = request.args.get('status')  # 'unread', 'read', or None for all
    conn = get_db()
    c = conn.cursor()
    if status:
        c.execute(
            'SELECT * FROM notifications WHERE status = ? ORDER BY created_at DESC LIMIT 50',
            (status,)
        )
    else:
        c.execute('SELECT * FROM notifications ORDER BY created_at DESC LIMIT 50')
    notifications = [dict(n) for n in c.fetchall()]
    conn.close()
    return jsonify({'notifications': notifications})


@app.route('/api/notifications/unread-count', methods=['GET'])
def unread_count():
    """Return the count of unread notifications."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM notifications WHERE status = 'unread'")
    count = c.fetchone()[0]
    conn.close()
    return jsonify({'unread_count': count})


@app.route('/api/notifications/<int:notification_id>/read', methods=['PUT'])
def mark_notification_read(notification_id):
    """Mark a single notification as read."""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id FROM notifications WHERE id = ?', (notification_id,))
    if c.fetchone() is None:
        conn.close()
        return jsonify({'error': 'Notification not found'}), 404
    c.execute("UPDATE notifications SET status = 'read' WHERE id = ?", (notification_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/notifications/read-all', methods=['PUT'])
def mark_all_notifications_read():
    """Mark all unread notifications as read."""
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE notifications SET status = 'read' WHERE status = 'unread'")
    conn.commit()
    conn.close()
    return jsonify({'success': True})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)