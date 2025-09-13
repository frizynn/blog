# Juan Lebrero's Blog

A modern, minimal blog built with Hugo and deployed on GitHub Pages.

## 🚀 Quick Start

### Development
```bash
# Start local development server
hugo server --minify --buildDrafts
```

### Deployment
```bash
# Deploy to production
./deploy.sh
```

## 📁 Project Structure

```
juanlebrero-blog/
├── content/           # Blog posts and pages
├── static/           # Static assets (images, CNAME, etc.)
├── themes/           # Hugo themes
├── hugo.toml         # Hugo configuration
├── deploy.sh         # Deployment script
└── frizynn.github.io/ # GitHub Pages submodule
```

## 🛠️ Technology Stack

- **Static Site Generator**: Hugo
- **Theme**: hugo-brewm
- **Hosting**: GitHub Pages
- **Domain**: juanlebrero.com

## 📝 Writing Posts

### Quick Start
```bash
# Create new post with folder structure
./new-post.sh "Your Post Title"
```

### Manual Creation
1. Create new post folder in `content/en/post/your-post-slug/`
2. Add `index.md` with proper frontmatter:
   ```yaml
   ---
   date: 2025-01-13T16:37:45-03:00
   draft: false
   title: 'Your Post Title'
   type: post
   ---
   ```
3. Add images in `content/en/post/your-post-slug/images/`

## 🔄 Workflow

1. **Edit content** → Make changes to posts, config, etc.
2. **Update source** → `git add . && git commit -m "message" && git push`
3. **Deploy site** → `./deploy.sh`

## 🌐 Live Sites

- **GitHub Pages**: https://frizynn.github.io
- **Custom Domain**: https://juanlebrero.com

---

*Built with ❤️ using Hugo and GitHub Pages*
