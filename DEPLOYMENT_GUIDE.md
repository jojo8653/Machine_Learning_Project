# 🚀 Team Access Deployment Guide

## Quick Setup for Team Access

### Option 1: Instant Sharing with ngrok (5 minutes)

1. **Open a new PowerShell window** (to use newly installed tools)
2. **Navigate to your project:**
   ```powershell
   cd "C:\Users\joann\OneDrive\Documents\Cornell\ML"
   ```

3. **Start Streamlit:**
   ```powershell
   streamlit run chatbot_interface.py
   ```

4. **Open ANOTHER PowerShell window and run:**
   ```powershell
   ngrok http 8501
   ```

5. **Copy the public URL** (looks like `https://xyz123.ngrok-free.app`)
6. **Share this URL with your team** - works from anywhere! 🌍

**Pros:** ✅ Instant access, ✅ No setup required
**Cons:** ❌ URL changes each restart, ❌ Limited bandwidth

---

### Option 2: Permanent Deployment (Streamlit Community Cloud)

#### Step 1: Set Up GitHub Repository

1. **Restart PowerShell** (to use Git)
2. **Navigate to project:**
   ```powershell
   cd "C:\Users\joann\OneDrive\Documents\Cornell\ML"
   ```

3. **Initialize Git repository:**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: AI Trading Chatbot with VADER sentiment"
   ```

4. **Create GitHub repository:**
   - Go to [github.com](https://github.com) and create account if needed
   - Click "New repository"
   - Name: `ai-trading-chatbot`
   - Make it Public
   - Don't initialize with README (we have one)

5. **Connect and push:**
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/ai-trading-chatbot.git
   git branch -M main
   git push -u origin main
   ```

#### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub account**
3. **Click "New app"**
4. **Select your repository:** `YOUR_USERNAME/ai-trading-chatbot`
5. **Main file path:** `chatbot_interface.py`
6. **Click "Deploy"**

**Result:** Your team gets a permanent URL like:
`https://your-app-name.streamlit.app` 🎉

---

## 📋 Pre-Deployment Checklist

✅ **Requirements.txt** - Already included  
✅ **Rate limiting protection** - Already implemented  
✅ **Error handling** - Already added  
✅ **Synthetic data fallback** - Already working  

## 🔧 Environment Variables (Optional)

If you want to add NewsAPI or other services:

1. **In Streamlit Cloud:** Go to App Settings → Secrets
2. **Add:**
   ```toml
   NEWS_API_KEY = "your_newsapi_key_here"
   ```

## 🎯 Team Access URLs

### Immediate (ngrok):
- **URL Format:** `https://xyz123.ngrok-free.app`
- **Access:** Anyone with the link
- **Duration:** While your computer is running

### Permanent (Streamlit Cloud):
- **URL Format:** `https://ai-trading-chatbot.streamlit.app`
- **Access:** 24/7 public access
- **Cost:** FREE

## 🛡️ Security Notes

- **Ngrok:** Temporary URLs, relatively secure
- **Streamlit Cloud:** Public by default, but no sensitive data exposed
- **API Keys:** Keep private, use Streamlit secrets for production

## 📞 Support

If you encounter issues:
1. Check the deployment logs in Streamlit Cloud
2. Verify all files are committed to GitHub
3. Ensure requirements.txt includes all dependencies
4. Test locally first with `streamlit run chatbot_interface.py`

---

**Quick Start:** Use ngrok for immediate team access!  
**Production:** Deploy to Streamlit Cloud for permanent 24/7 access! 