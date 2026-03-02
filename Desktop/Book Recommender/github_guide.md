# GitHub Upload Guide: Semantic Book Recommender

## Repository Name: Semantic-Book-Recommender

## Step-by-Step Instructions

### Step 1: Create GitHub Repository
1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon → "New repository"
3. Repository name: `Semantic-Book-Recommender`
4. Description: "A semantic book recommendation system using Sentence Transformers and zero-shot classification"
5. Keep it **Public** (for academic sharing)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### Step 2: Get Repository URL
After creation, copy the repository URL from the green "Code" button:
```
https://github.com/YOUR_USERNAME/Semantic-Book-Recommender.git
```

### Step 3: Connect Local Repository to GitHub
Open PowerShell/Command Prompt and run:

```bash
cd "c:\Users\noa3\Desktop\Book Recommender"
git remote add origin https://github.com/YOUR_USERNAME/Semantic-Book-Recommender.git
git branch -M main
git push -u origin main
```

### Step 4: Verify Upload
1. Go back to your GitHub repository page
2. Refresh the page - you should see all your files
3. The repository should show:
   - 20 files
   - MIT License badge
   - Description from README

### Step 5: Enable GitHub Pages (Optional)
For a live demo page:
1. Go to repository Settings → Pages
2. Source: "Deploy from a branch"
3. Branch: "main" → "/ (root)"
4. Save
5. Wait 2-3 minutes, then access: `https://YOUR_USERNAME.github.io/Semantic-Book-Recommender`

## Troubleshooting

### If you get authentication errors:
```bash
# Generate a Personal Access Token on GitHub:
# Settings → Developer settings → Personal access tokens → Generate new token
# Give it 'repo' permissions

git push -u origin main
# When prompted for password, use your Personal Access Token (not GitHub password)
```

### If repository already exists:
```bash
# If you need to change the remote URL:
git remote set-url origin https://github.com/YOUR_USERNAME/Semantic-Book-Recommender.git
git push -u origin main
```

### If you get "fatal: remote origin already exists":
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/Semantic-Book-Recommender.git
git push -u origin main
```

## What Gets Uploaded

✅ **Core Files:**
- `Nigel_Andati_final_project_colab.ipynb` - Main submission notebook
- `create_embeddings_subset.py` - Standalone embedding generator
- `local_development.py` - Local development runner
- `README.md` - Professional documentation
- `requirements.txt` - Dependencies
- `LICENSE` - MIT License
- `.gitignore` - Git rules

✅ **Modular Codebase:**
- `config.py` - Configuration
- `data/pipeline.py` - Data processing
- `embeddings/store.py` - Vector management
- `retrieval/retriever.py` - Search system
- `classification/` - AI analysis
- `ui/` - Web interface

✅ **GitHub Integration:**
- `.github/workflows/test.yml` - CI/CD pipeline

❌ **Excluded (as intended):**
- Virtual environments (`venv/`)
- Large data files (`*.npy`, `*.zip`)
- Temporary files (`__pycache__/`, `.gradio/`)

## Repository Features

After upload, your repository will have:
- ⭐ Professional README with badges and instructions
- 🔧 Automated testing on multiple Python versions
- 📄 MIT License for open source
- 🤝 Contributing guidelines
- 📊 Project structure documentation
- 🎯 Academic AI disclosure section

## Demo Instructions

Tell viewers to:
1. **For Grading**: Use the Colab notebook
2. **For Local Development**: Follow README installation steps
3. **Expected Results**: 7,000+ books with semantic search and classification

## Success Checklist

- [ ] Repository created on GitHub with name: `Semantic-Book-Recommender`
- [ ] All 20 files uploaded successfully
- [ ] README renders properly
- [ ] License badge visible
- [ ] Repository is public
- [ ] CI/CD workflow passes (green checkmark)

🎉 **Your Semantic Book Recommender is now live on GitHub!**