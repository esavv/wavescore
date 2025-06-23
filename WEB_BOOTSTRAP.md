## Project Summary

I want to build a simple web app for my surf score prediction project. I'm imagining a simple React-based web app that allows the user to submit a surf video, calls my API to analyze the video, renders the APIs SSE responses, and displays the final results to the user. I want it to look slick and modern, with a minimalist aesthetic. I also want to build this iteratively, and deploy it to a live, publicly available website as soon as possible... later today if I can, even if all/most functionality isn't ready yet.

## Tech Stack Recommendation

- **Frontend**: `React` + `Vite`  
  _Fast build times, ideal for minimal SPAs. Option to switch to Next.js later if needed._
  
- **Styling**: `Tailwind CSS`  
  _Modern, minimalist, utility-first styling with fast iteration._

- **API Integration**: Native `fetch` with `EventSource`  
  _Support for Server-Sent Events (SSE) to stream analysis updates._

- **Hosting**: [**Vercel**](https://vercel.com)  
  _One-click deploy, automatic from GitHub, free tier is excellent._

- **Codebase**: Own and version from Day 1  
  _Use GitHub + Cursor (or your preferred IDE)_

---

## Fast Path to Launch Today

### 1. Scaffold Your App with Vite + React

```bash
npm init vite@latest web
cd web
npm install
```

### 2. Add Tailwind CSS

```bash
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p
```

- In `tailwind.config.js`, set the `content` array:
```js
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

- In `src/index.css`, replace all content with:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 3. Add CSS Linting with Stylelint (Tailwind Compatible)

```bash
npm install -D stylelint stylelint-config-standard stylelint-config-tailwindcss
```

- In your project root, create a `stylelint.config.cjs` file with:
```js
module.exports = {
  extends: [
    "stylelint-config-standard",
    "stylelint-config-tailwindcss"
  ],
  rules: {}
}
```

### 4. Create Minimal UI in `App.tsx`

Include:
- File upload input
- Submit button
- Stub area to stream analysis results (hook up SSE later)
- Tailwind styles for layout & polish

### 5. Run Locally

```bash
npm run dev
```

This starts the development server (usually at `http://localhost:5173`). Your app will hot-reload as you make changes.

### 6. Deploy with Vercel

- Go to [https://vercel.com](https://vercel.com)
- Sign in with GitHub
- Import your wavescore repo

**For Monorepo Setup:**

Since we're building this as part of a larger wavescore project, configure Vercel to deploy only the web subdirectory:

- During Vercel setup, set **Root Directory** to `web/`
- Set **Build Command** to `npm run build`
- Set **Output Directory** to `dist` (Vite default)

Your project structure:
```
wavescore/
  ├── api/
  ├── mobile/
  ├── web/         👈 Vercel will build from here
  ├── README.md
  └── .git/
```

Vercel will only build and deploy from the `web/` folder, ignoring the rest of your monorepo.

- Click **Deploy**
