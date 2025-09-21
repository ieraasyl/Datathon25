from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class RawComment(Base):
    """Модель для сырых комментариев из соцсетей"""
    __tablename__ = 'raw_comments'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)  # Источник файла
    post_url = Column(Text, nullable=True)
    text = Column(Text, nullable=False)
    username = Column(String(255), nullable=True)
    like_count = Column(Integer, default=0)
    created_at_utc = Column(DateTime, nullable=True)
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<RawComment(id={self.id}, username='{self.username}', text='{self.text[:50]}...')>"

class AnalyzedComment(Base):
    """Модель для проанализированных комментариев"""
    __tablename__ = 'analyzed_comments'
    
    id = Column(Integer, primary_key=True)
    raw_comment_id = Column(Integer, nullable=False)  # Ссылка на оригинальный комментарий
    
    # Исходные данные
    filename = Column(String(255), nullable=False)
    platform = Column(String(50), nullable=False)  # instagram, vk
    brand = Column(String(50), nullable=False)  # altel, tele2
    post_url = Column(Text)
    text = Column(Text, nullable=False)
    username = Column(String(255))
    like_count = Column(Integer, default=0)
    created_at_utc = Column(DateTime)
    
    # Результаты анализа
    comment_types = Column(JSON)  # ['complaint', 'question'] etc
    sentiment = Column(String(50))  # very_negative, negative, neutral, positive, very_positive
    key_idea = Column(Text)
    swear_words = Column(JSON)  # Список нецензурной лексики
    language = Column(String(10))  # ru, kz
    
    # Метаданные
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analysis_model = Column(String(100), default='gemini-2.5-flash')
    
    def __repr__(self):
        return f"<AnalyzedComment(id={self.id}, sentiment='{self.sentiment}', types={self.comment_types})>"

class AnalysisStats(Base):
    """Модель для хранения статистики анализа"""
    __tablename__ = 'analysis_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    
    # Общая статистика
    total_comments = Column(Integer, default=0)
    total_posts = Column(Integer, default=0)
    
    # Статистика по брендам
    altel_comments = Column(Integer, default=0)
    tele2_comments = Column(Integer, default=0)
    
    # Статистика по платформам
    instagram_comments = Column(Integer, default=0)
    vk_comments = Column(Integer, default=0)
    
    # Статистика по языкам
    russian_comments = Column(Integer, default=0)
    kazakh_comments = Column(Integer, default=0)
    
    # Статистика по типам
    questions = Column(Integer, default=0)
    complaints = Column(Integer, default=0)
    reviews = Column(Integer, default=0)
    thanks = Column(Integer, default=0)
    spam = Column(Integer, default=0)
    
    # Статистика по настроению
    very_negative = Column(Integer, default=0)
    negative = Column(Integer, default=0)
    neutral = Column(Integer, default=0)
    positive = Column(Integer, default=0)
    very_positive = Column(Integer, default=0)
    
    # Статистика по нецензурной лексике
    comments_with_swear = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<AnalysisStats(date='{self.date}', total_comments={self.total_comments})>"

# Настройка подключения к базе данных
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///comments_analysis.db')

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Создание таблиц в базе данных"""
    Base.metadata.create_all(bind=engine)
    print("Таблицы созданы успешно")

def get_db_session():
    """Получение сессии базы данных"""
    return SessionLocal()

if __name__ == "__main__":
    # Создаем таблицы при запуске
    create_tables()