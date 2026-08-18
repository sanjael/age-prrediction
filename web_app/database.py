import os
import datetime
import urllib.parse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class BiometricAuditLog(Base):
    __tablename__ = "cts_agevision_audit"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_id = Column(String(50), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image_name = Column(String(255))
    image_base64 = Column(Text(length=4294967295)) # LongText in MySQL to store full image thumbnail
    predicted_age = Column(Float, nullable=False)
    confidence_range = Column(String(50))
    age_category = Column(String(50))
    model_dex_pred = Column(Float)
    model_hybrid_pred = Column(Float)
    model_convnext_pred = Column(Float)
    client_use_case = Column(String(50))
    latency_ms = Column(Integer)

# Safe URL-encoded password for MySQL: 'san@2005' -> 'san%402005'
encoded_pwd = urllib.parse.quote_plus("san@2005")
MYSQL_URL = f"mysql+pymysql://root:{encoded_pwd}@localhost:3306/cts_agevision"
SQLITE_URL = "sqlite:///cts_agevision.db"

engine = None
try:
    test_engine = create_engine(MYSQL_URL, pool_pre_ping=True)
    with test_engine.connect() as conn:
        pass
    engine = test_engine
    print("[+] Database: Connected to MySQL (root:san@2005 @ localhost:3306/cts_agevision) successfully!")
except Exception as e:
    print(f"[*] MySQL error ({e}), falling back to SQLite.")
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("[+] MySQL Table `cts_agevision_audit` created/verified successfully with image storage!")

def log_scan(scan_id, image_name, image_base64, predicted_age, confidence_range, age_category, 
             m_dex, m_hyb, m_cnx, client_use_case, latency_ms):
    db = SessionLocal()
    try:
        record = BiometricAuditLog(
            scan_id=scan_id,
            image_name=image_name,
            image_base64=image_base64,
            predicted_age=predicted_age,
            confidence_range=confidence_range,
            age_category=age_category,
            model_dex_pred=m_dex,
            model_hybrid_pred=m_hyb,
            model_convnext_pred=m_cnx,
            client_use_case=client_use_case,
            latency_ms=latency_ms
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    except Exception as err:
        db.rollback()
        print(f"[-] Database Insert Error: {err}")
        return None
    finally:
        db.close()

def get_recent_logs(limit=25):
    db = SessionLocal()
    try:
        records = db.query(BiometricAuditLog).order_by(BiometricAuditLog.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "scan_id": r.scan_id,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
                "image_name": r.image_name,
                "image_base64": r.image_base64,
                "predicted_age": r.predicted_age,
                "confidence_range": r.confidence_range,
                "age_category": r.age_category,
                "model_dex_pred": r.model_dex_pred,
                "model_hybrid_pred": r.model_hybrid_pred,
                "model_convnext_pred": r.model_convnext_pred,
                "client_use_case": r.client_use_case,
                "latency_ms": r.latency_ms
            }
            for r in records
        ]
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
