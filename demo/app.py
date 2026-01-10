"""
Gradio Demo for Recommendation System
Modern UI with theme toggle, model selection, and image display
"""

import os
import sys
from pathlib import Path

# Change to project root and add to path
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Hybrid_LightGCN_TFIDF"))

import gradio as gr
import random

# Import Hybrid Recommender
from Hybrid_LightGCN_TFIDF.model import load_hybrid_recommender
from Hybrid_LightGCN_TFIDF.config import config


# Custom CSS for modern rounded UI
CUSTOM_CSS = """
/* Modern rounded corners */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gr-button {
    border-radius: 16px !important;
    font-weight: 500 !important;
}

.gr-input, .gr-textbox, .gr-dropdown {
    border-radius: 12px !important;
}

.gr-panel {
    border-radius: 20px !important;
}

.gr-box {
    border-radius: 16px !important;
}

.gr-form {
    border-radius: 16px !important;
}

/* Card style for recommendations */
.recommendation-card {
    background: var(--background-fill-secondary);
    border-radius: 16px;
    padding: 16px;
    margin: 8px 0;
    transition: transform 0.2s, box-shadow 0.2s;
}

.recommendation-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.product-image {
    border-radius: 12px;
    width: 80px;
    height: 80px;
    object-fit: cover;
}

.product-info {
    display: flex;
    gap: 16px;
    align-items: center;
}

.product-details h4 {
    margin: 0 0 4px 0;
    font-size: 14px;
}

.product-meta {
    color: var(--body-text-color-subdued);
    font-size: 12px;
}

.score-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* History table styling */
.history-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 8px;
}

.history-table th {
    text-align: left;
    padding: 12px;
    background: var(--background-fill-secondary);
    border-radius: 12px 12px 0 0;
}

.history-table td {
    padding: 12px;
    background: var(--background-fill-primary);
}

.history-table tr td:first-child {
    border-radius: 12px 0 0 12px;
}

.history-table tr td:last-child {
    border-radius: 0 12px 12px 0;
}

/* Method comparison cards */
.method-card {
    border-radius: 16px;
    padding: 16px;
    background: var(--background-fill-secondary);
}

.method-title {
    font-weight: 600;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border-color-primary);
}

/* Tab styling */
.gr-tab-nav {
    border-radius: 16px !important;
}

.gr-tab-nav button {
    border-radius: 12px !important;
}
"""


class RecommenderDemo:
    """Wrapper for Gradio interface with enhanced features"""
    
    def __init__(self):
        print("Loading Hybrid Recommender...")
        self.hybrid = load_hybrid_recommender(alpha=config.hybrid.alpha, device="cuda")
        self.data = self.hybrid.lightgcn_data
        print(f"Ready! Users: {self.data.num_users:,} | Items: {self.data.num_items:,}")
    
    def get_sample_users(self, n: int = 20):
        return random.sample(list(self.data.user2idx.keys()), min(n, self.data.num_users))
    
    def get_sample_items(self, n: int = 20):
        return random.sample(list(self.data.item2idx.keys()), min(n, self.data.num_items))
    
    def format_product_card(self, rec: dict, rank: int) -> str:
        """Format a single product as modern card HTML"""
        name = rec.get('name', 'Unknown')[:70]
        category = rec.get('category', 'N/A')
        price = rec.get('price', 0)
        score = rec.get('score', rec.get('similarity', 0))
        image_url = rec.get('image', 'https://via.placeholder.com/80x80?text=No+Image')
        
        return f"""
        <div class="recommendation-card">
            <div class="product-info">
                <img src="{image_url}" class="product-image" onerror="this.src='https://via.placeholder.com/80x80?text=No+Image'">
                <div class="product-details" style="flex: 1;">
                    <h4>#{rank}. {name}</h4>
                    <p class="product-meta">{category}</p>
                    <p class="product-meta" style="color: #e53e3e; font-weight: 600;">{price:,.0f} VND</p>
                </div>
                <span class="score-badge">{score:.4f}</span>
            </div>
        </div>
        """
    
    def format_history(self, user_idx: int) -> str:
        """Format user history as styled HTML table"""
        history = self.data.user_pos_items.get(user_idx, set())
        if not history:
            return "<p style='text-align: center; color: gray;'>No history available</p>"
        
        rows = []
        for i, item_idx in enumerate(list(history)[:8], 1):
            item_id = self.data.idx2item.get(item_idx)
            info = self.data.item_info.get(item_id, {})
            name = info.get('name', 'Unknown')[:50]
            category = info.get('category', 'N/A')
            price = info.get('price', 0)
            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{name}</td>
                    <td>{category}</td>
                    <td style="color: #e53e3e; font-weight: 500;">{price:,.0f} VND</td>
                </tr>
            """)
        
        extra = len(history) - 8
        if extra > 0:
            rows.append(f"<tr><td colspan='4' style='text-align: center; font-style: italic;'>... and {extra} more items</td></tr>")
        
        return f"""
        <table class="history-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Product</th>
                    <th>Category</th>
                    <th>Price</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        """
    
    def format_recommendations(self, recs: list) -> str:
        """Format recommendations as modern cards"""
        if not recs:
            return "<p style='text-align: center; color: gray;'>No recommendations available</p>"
        
        cards = [self.format_product_card(rec, i) for i, rec in enumerate(recs, 1)]
        return f"<div style='display: flex; flex-direction: column; gap: 8px;'>{''.join(cards)}</div>"
    
    def recommend_user(self, user_id: str, model: str, alpha: float, top_k: int):
        """Get recommendations for a user with selected model"""
        try:
            user_id = int(user_id)
        except:
            return "Invalid user ID", ""
        
        if user_id not in self.data.user2idx:
            sample = list(self.data.user2idx.keys())[:5]
            return f"<p>User not found. Try: {sample}</p>", ""
        
        user_idx = self.data.user2idx[user_id]
        user_history = self.data.user_pos_items.get(user_idx, set())
        
        # Set alpha based on model selection
        if model == "LightGCN":
            self.hybrid.set_alpha(1.0)
        elif model == "TF-IDF":
            self.hybrid.set_alpha(0.0)
        else:  # Hybrid
            self.hybrid.set_alpha(alpha)
        
        recs = self.hybrid.recommend_with_details(user_idx, user_history, top_k=int(top_k))
        
        return self.format_history(user_idx), self.format_recommendations(recs)
    
    def recommend_similar_items(self, item_id: str, top_k: int):
        """Get similar items (content-based)"""
        try:
            item_id = int(item_id)
        except:
            return "<p>Invalid item ID</p>", ""
        
        if item_id not in self.data.item2idx:
            sample = list(self.data.item2idx.keys())[:5]
            return f"<p>Item not found. Try: {sample}</p>", ""
        
        item_idx = self.data.item2idx[item_id]
        info = self.data.item_info.get(item_id, {})
        
        # Get TF-IDF based similar items
        self.hybrid.set_alpha(0.0)  # Pure content-based
        
        # Create a fake user with just this item
        fake_history = {item_idx}
        recs = self.hybrid.recommend_with_details(0, fake_history, top_k=int(top_k))
        
        # Query item info
        query_html = f"""
        <div class="recommendation-card" style="border: 2px solid #667eea;">
            <div class="product-info">
                <div class="product-details" style="flex: 1;">
                    <h4>Query: {info.get('name', 'Unknown')[:60]}</h4>
                    <p class="product-meta">{info.get('category', 'N/A')}</p>
                    <p class="product-meta" style="color: #e53e3e; font-weight: 600;">{info.get('price', 0):,.0f} VND</p>
                </div>
            </div>
        </div>
        """
        
        return query_html, self.format_recommendations(recs)
    
    def compare_methods(self, user_id: str, top_k: int):
        """Compare all methods side by side"""
        try:
            user_id = int(user_id)
        except:
            return "", "", ""
        
        if user_id not in self.data.user2idx:
            return "<p>User not found</p>", "<p>User not found</p>", "<p>User not found</p>"
        
        user_idx = self.data.user2idx[user_id]
        user_history = self.data.user_pos_items.get(user_idx, set())
        
        results = {}
        for alpha, name in [(1.0, "LightGCN"), (0.0, "TF-IDF"), (0.8, "Hybrid")]:
            self.hybrid.set_alpha(alpha)
            recs = self.hybrid.recommend_with_details(user_idx, user_history, top_k=int(top_k))
            results[name] = self.format_recommendations(recs)
        
        return results["LightGCN"], results["TF-IDF"], results["Hybrid"]
    
    def random_user(self):
        return str(random.choice(list(self.data.user2idx.keys())))
    
    def random_item(self):
        return str(random.choice(list(self.data.item2idx.keys())))


def create_app():
    """Create enhanced Gradio app with modern UI"""
    demo_rec = RecommenderDemo()
    
    # Theme with rounded corners
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)",
        block_radius="16px",
        container_radius="20px",
        input_radius="12px",
    )
    
    with gr.Blocks(title="Recommendation System Demo") as app:
        
        # Header
        gr.Markdown("""
        # 🛒 Recommendation System Demo
        ### Hybrid LightGCN + TF-IDF for Vietnamese E-commerce
        
        Dataset: [Qhuy204/TikDataset](https://huggingface.co/datasets/Qhuy204/TikDataset) | 
        Source: [tiki.vn](https://tiki.vn)
        """)
        
        # Tab 1: User Recommendations
        with gr.Tab("👤 User Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    user_input = gr.Textbox(
                        label="User ID",
                        placeholder="Enter user ID or click Random...",
                        elem_classes=["gr-textbox"]
                    )
                    random_btn = gr.Button("🎲 Random User", variant="secondary")
                    
                    model_select = gr.Dropdown(
                        choices=["Hybrid", "LightGCN", "TF-IDF"],
                        value="Hybrid",
                        label="Model"
                    )
                    
                    alpha_slider = gr.Slider(
                        0, 1, value=0.8, step=0.05,
                        label="Alpha (only for Hybrid)",
                        info="1.0 = LightGCN, 0.0 = TF-IDF"
                    )
                    
                    topk_slider = gr.Slider(5, 15, value=10, step=1, label="Top-K")
                    
                    submit_btn = gr.Button("🔍 Get Recommendations", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### User History")
                    history_output = gr.HTML()
                    
                    gr.Markdown("### Recommendations")
                    recs_output = gr.HTML()
            
            random_btn.click(demo_rec.random_user, outputs=user_input)
            submit_btn.click(
                demo_rec.recommend_user,
                inputs=[user_input, model_select, alpha_slider, topk_slider],
                outputs=[history_output, recs_output]
            )
        
        # Tab 2: Similar Items
        with gr.Tab("📦 Similar Items"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Find Similar Products")
                    item_input = gr.Textbox(
                        label="Item ID",
                        placeholder="Enter item ID..."
                    )
                    random_item_btn = gr.Button("🎲 Random Item", variant="secondary")
                    item_topk = gr.Slider(5, 15, value=10, step=1, label="Top-K")
                    item_submit = gr.Button("🔍 Find Similar", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Query Item")
                    query_output = gr.HTML()
                    
                    gr.Markdown("### Similar Items")
                    similar_output = gr.HTML()
            
            random_item_btn.click(demo_rec.random_item, outputs=item_input)
            item_submit.click(
                demo_rec.recommend_similar_items,
                inputs=[item_input, item_topk],
                outputs=[query_output, similar_output]
            )
        
        # Tab 3: Compare Methods
        with gr.Tab("⚖️ Compare Methods"):
            with gr.Row():
                compare_user = gr.Textbox(label="User ID", placeholder="Enter user ID...")
                compare_random = gr.Button("🎲 Random", variant="secondary")
                compare_topk = gr.Slider(3, 8, value=5, step=1, label="Top-K")
                compare_btn = gr.Button("⚖️ Compare All Methods", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### LightGCN (α=1.0)")
                    lgcn_output = gr.HTML()
                with gr.Column():
                    gr.Markdown("### TF-IDF (α=0.0)")
                    tfidf_output = gr.HTML()
                with gr.Column():
                    gr.Markdown("### Hybrid (α=0.8)")
                    hybrid_output = gr.HTML()
            
            compare_random.click(demo_rec.random_user, outputs=compare_user)
            compare_btn.click(
                demo_rec.compare_methods,
                inputs=[compare_user, compare_topk],
                outputs=[lgcn_output, tfidf_output, hybrid_output]
            )
        
        # Footer
        gr.Markdown("""
        ---
        **Models**: LightGCN (SIGIR 2020) + TF-IDF | **Best α**: 0.8 | **HR@10**: 29.80%
        """)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
