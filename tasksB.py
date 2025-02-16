
import os

def B12(filepath):
    if filepath.startswith('/data'):
       
        return True
    else:
        return False

def B3(url, save_path):
    if not B12(save_path):
        return None
    import requests
    response = requests.get(url)
    with open(save_path, 'w') as file:
        file.write(response.text)

def B5(db_path, query, output_filename):
    if not B12(db_path):
        return None
    import sqlite3, duckdb
    conn = sqlite3.connect(db_path) if db_path.endswith('.db') else duckdb.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()
    with open(output_filename, 'w') as file:
        file.write(str(result))
    return result

def B6(url, output_filename):
    import requests
    result = requests.get(url).text
    with open(output_filename, 'w') as file:
        file.write(str(result))

def B7(image_path, output_path, resize=None):
    from PIL import Image
    if not B12(image_path):
        return None
    if not B12(output_path):
        return None
    img = Image.open(image_path)
    if resize:
        img = img.resize(resize)
    img.save(output_path)


def B9(md_path, output_path):
    import markdown
    if not B12(md_path):
        return None
    if not B12(output_path):
        return None
    with open(md_path, 'r') as file:
        html = markdown.markdown(file.read())
    with open(output_path, 'w') as file:
        file.write(html)


