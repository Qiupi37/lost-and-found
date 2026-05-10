import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sqlite3
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer

# ------------------ 初始化应用 ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 初始化数据库（如果不存在则创建）
def init_db():
    conn = sqlite3.connect('items.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    number TEXT UNIQUE,
                    description TEXT,
                    location TEXT,
                    images TEXT
                )''')
    conn.commit()
    conn.close()

init_db()   # 应用启动时自动执行

# 加载AI模型（在 Render 上会自动从 HuggingFace 下载，速度很快）
print("正在加载AI模型（请耐心等待，下载中...）")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("模型就绪！")


# ------------------ 页面路由 ------------------

@app.route('/')
def index():
    return redirect(url_for('search'))

@app.route('/add', methods=['GET', 'POST'])
def add_item():
    if request.method == 'POST':
        number = request.form['number']
        description = request.form['description']
        location = request.form['location']
        uploaded_files = request.files.getlist('images')
        image_paths = []
        for file in uploaded_files:
            if file.filename:
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                image_paths.append(filename)
        images_str = ','.join(image_paths)

        conn = sqlite3.connect('items.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO items (number, description, location, images) VALUES (?, ?, ?, ?)",
                      (number, description, location, images_str))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return "❌ 该编号已存在，请使用唯一编号！"
        conn.close()
        return redirect(url_for('add_item'))
    return render_template('add_item.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    results = None
    query = ''
    if request.method == 'POST':
        query = request.form['query']
        conn = sqlite3.connect('items.db')
        c = conn.cursor()
        c.execute("SELECT number, description, location, images FROM items")
        items = c.fetchall()
        conn.close()

        if not items:
            return render_template('search.html', results=[], query=query, message="数据库里还没有物品，请先添加。")

        descriptions = [item[1] for item in items]
        all_texts = [query] + descriptions
        embeddings = model.encode(all_texts)
        query_embedding = embeddings[0]
        item_embeddings = embeddings[1:]

        similarities = np.dot(item_embeddings, query_embedding) / (
            np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:5]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:
                number, desc, location, imgs = items[idx]
                img_list = imgs.split(',') if imgs else []
                results.append({
                    'number': number,
                    'description': desc,
                    'location': location,
                    'images': img_list,
                    'score': round(float(similarities[idx]), 2)
                })
        message = None if results else "没找到特别匹配的物品，试试换种方式描述。"
        return render_template('search.html', results=results, query=query, message=message)
    return render_template('search.html', results=None, query='', message=None)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    conn = sqlite3.connect('items.db')
    c = conn.cursor()
    if request.method == 'POST':
        item_id = request.form.get('delete_id')
        if item_id:
            c.execute("DELETE FROM items WHERE id = ?", (item_id,))
            conn.commit()
        conn.close()
        return redirect(url_for('admin'))
    c.execute("SELECT id, number, description, location, images FROM items ORDER BY number")
    items = c.fetchall()
    conn.close()
    return render_template('admin.html', items=items)


# ------------------ 本地启动（Render 会用 gunicorn，不会执行这里） ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)