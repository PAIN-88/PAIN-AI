# PAIN

A multi-model AI chat app built with **Chainlit** and **LangGraph**, using models from **OpenRouter**. It has user accounts, saved chat history, and image and document uploads.

## Features

- **Multiple models:** switch between models during a chat. The conversation history is shared, so the new model knows what was said before.
- **File uploads:** send images, or documents (PDF, DOCX, XLSX, TXT, CSV, JSON and code files). Uploaded documents stay available for follow-up questions in the same chat.
- **User accounts:** anyone can sign up at `/signup` and log in. Passwords are stored hashed.
- **Saved chats:** every user has a private list of past chats in the sidebar and can reopen them, with their history and documents restored.
- **Custom branding:** PAIN name, logo and favicon.

## Tech stack

Python, Chainlit, LangGraph, LangChain (OpenAI-compatible client), OpenRouter, SQLite.

## Getting started

1. Clone the repository and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project folder:

   ```
   OPENROUTER_API_KEY=your-openrouter-key
   CHAINLIT_AUTH_SECRET=your-generated-secret
   ```

   Generate the secret with `chainlit create-secret`.

3. Run the app:

   ```bash
   chainlit run app.py -w
   ```

4. Open `http://localhost:8000/signup`, create an account, then log in.

## Project structure

```
├── app.py                  # Main application
├── requirements.txt
├── public/
│   ├── logo_dark.png       # Login/header logo (dark theme)
│   ├── logo_light.png      # Login/header logo (light theme)
│   ├── favicon.png         # Browser tab icon
│   └── signup-link.js      # Adds "Create an account" link to the login page
└── .chainlit/config.toml   # Chainlit settings (app name PAIN, custom JS)
```

Created automatically when the app runs: `chat_history.db` (accounts and chats) and `saved_files/` (copies of uploaded files).

## Notes

- The free OpenRouter models have small context windows. Very large documents may fail; lower `MAX_CHARS_PER_DOC` and `MAX_CHARS_TOTAL` in `app.py` if that happens.
- Images are only sent with the message they were attached to. Documents stay active for the whole chat.
- On hosts with temporary disks (for example free Hugging Face Spaces), `chat_history.db` and `saved_files/` are erased when the app restarts. Use persistent storage if you need to keep accounts and chats.
- Sign-up is open to anyone who has the link, and all users share one OpenRouter API key.
