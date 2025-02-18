import sqlite3
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


SQLITE_DB_PATH = './food_orders.db'


def get_order_status(order_id: int):
    """Retrieve the status of an existing order."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT status FROM food_orders WHERE id = ?
    """, (order_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None


def cancel_order(order_id: int, phone_number: str):
    """Mark an order as cancelled."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE food_orders SET status='canceled' WHERE id=? and person_phone_number=?
    """, (order_id, phone_number))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def comment_order(order_id: int, phon_number: str, comment: str):
    """Update food_orders for an order."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE food_orders SET comment=? WHERE id=? and person_phone_number=?
    """, (comment, order_id, phon_number))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def search_food_orders(food_name: str, restaurant_name: str):
    """Search food_orders for records related to a specific food and restaurant name."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM foods WHERE food_name LIKE ? OR restaurant_name LIKE ?
    """, (f"%{food_name}%", f"%{restaurant_name}%"))
    results = cursor.fetchall()
    conn.close()
    return results


def search_food_orders(food_name: str, restaurant_name: str = ""):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    if restaurant_name:
        query = """
            SELECT food_name, restaurant_name, price FROM foods 
            WHERE food_name LIKE ? AND restaurant_name LIKE ?
        """
        params = (f"%{food_name}%", f"%{restaurant_name}%")
    else:
        query = """
            SELECT food_name, restaurant_name, price FROM foods 
            WHERE food_name LIKE ?
        """
        params = (f"%{food_name}%",)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()
    return results


def load_key(key):
    load_dotenv()

    API_KEY = os.getenv(key)

    if not API_KEY:
        raise ValueError("API_KEY is missing! Check your .env file.")
    return API_KEY


def calculate_similarity(query_vector, document_vectors):
    distances = cosine_similarity([query_vector], document_vectors)
    return distances[0]
