import json
import os
import webbrowser
from config import Config

class HTMLVisualizer:
    def __init__(self):
        self.output_dir = Config.OUTPUT_DIR
        self.images_dir = "images" # Relative path for HTML
        self.annotations_path = os.path.join(self.output_dir, 'annotations.json')
        self.html_path = os.path.join(self.output_dir, 'index.html')

    def load_data(self):
        if not os.path.exists(self.annotations_path):
            print("Error: Annotations file not found.")
            return []
        with open(self.annotations_path, 'r') as f:
            return json.load(f)

    def generate_html(self):
        data = self.load_data()
        json_data = json.dumps(data)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Visualizer</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #f0f0f0; }}
        .controls {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; display: flex; align-items: center; gap: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; }}
        .card {{ background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-2px); }}
        .card img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .label {{ margin-top: 8px; font-weight: bold; color: #333; }}
        .filename {{ font-size: 0.8em; color: #666; word-break: break-all; }}
        button {{ padding: 8px 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        select {{ padding: 8px; border-radius: 4px; border: 1px solid #ccc; }}
        .pagination {{ display: flex; align-items: center; gap: 10px; margin-left: auto; }}
        .badge {{ padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }}
        .badge-inside {{ background: #e3f2fd; color: #0d47a1; }}
        .badge-outside {{ background: #ffebee; color: #c62828; }}
    </style>
</head>
<body>

    <div class="controls">
        <label>Filter: 
            <select id="filterSelect" onchange="updateView()">
                <option value="All">All</option>
                <option value="inside">Inside</option>
                <option value="outside">Outside</option>
            </select>
        </label>
        
        <div class="pagination">
            <span id="pageInfo">Page 1</span>
            <button onclick="prevPage()" id="btnPrev">Previous</button>
            <button onclick="nextPage()" id="btnNext">Next</button>
        </div>
    </div>

    <div id="grid" class="grid"></div>

    <script>
        const data = {json_data};
        const itemsPerPage = 20;
        let currentPage = 0;
        let filteredData = [];

        function init() {{
            updateFilter();
        }}

        function updateFilter() {{
            const filter = document.getElementById('filterSelect').value;
            if (filter === 'All') {{
                filteredData = data;
            }} else {{
                filteredData = data.filter(item => item.class === filter);
            }}
            currentPage = 0;
            render();
        }}

        function updateView() {{
            updateFilter();
        }}

        function render() {{
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            
            // Show all filtered items
            filteredData.forEach(item => {{
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <img src="{self.images_dir}/${{item.filename}}" alt="${{item.class}}" loading="lazy">
                    <div class="label"><span class="badge badge-${{item.class}}">${{item.class}}</span></div>
                    <div class="filename">${{item.filename}}</div>
                `;
                grid.appendChild(card);
            }});

            // Update Info
            document.getElementById('pageInfo').textContent = `Showing all ${{filteredData.length}} items`;
            document.getElementById('btnPrev').style.display = 'none';
            document.getElementById('btnNext').style.display = 'none';
        }}

        function nextPage() {{
            const totalPages = Math.ceil(filteredData.length / itemsPerPage);
            if (currentPage < totalPages - 1) {{
                currentPage++;
                render();
            }}
        }}

        function prevPage() {{
            if (currentPage > 0) {{
                currentPage--;
                render();
            }}
        }}

        init();
    </script>
</body>
</html>
        """
        
        with open(self.html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Generated gallery at {self.html_path}")
        return self.html_path

    def open_browser(self):
        path = self.generate_html()
        if path:
            url = f'file://{os.path.abspath(path)}'
            webbrowser.open(url)
            print(f"Opening {url}")

if __name__ == "__main__":
    viz = HTMLVisualizer()
    viz.open_browser()
