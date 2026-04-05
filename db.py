from app import app, db
from sqlalchemy import text

sql = """
CREATE TABLE IF NOT EXISTS cognitive_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    memory_score INTEGER DEFAULT 0,
    recall_score INTEGER DEFAULT 0,
    orientation_score INTEGER DEFAULT 0,
    visuospatial_score INTEGER DEFAULT 0,
    total_score INTEGER DEFAULT 0,
    answers_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
"""

with app.app_context():
    try:
        with db.engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()

        print("✅ cognitive_tests table created successfully!")

    except Exception as e:
        print("❌ Error creating table:", e)
