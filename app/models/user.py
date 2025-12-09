from ..extensions import db

class User(db.Model):
    __tablename__ = 'user'  # Убедитесь, что вы указали имя таблицы
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

    __table_args__ = {'extend_existing': True}  # Добавляем этот аргумент
