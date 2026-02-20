---
name: medium-to-jekyll
version: 2.0.0
description: |
  This skill should be used when the user asks to "convert Medium to Jekyll",
  "import Medium post", "Medium to markdown", "fetch Medium article",
  "convert blog post from Medium", or provides a Medium URL and wants to
  publish it to their Jekyll blog. Converts Medium articles to Jekyll-compatible
  markdown with proper front matter, image handling, and formatting.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - WebFetch
---

# Medium to Jekyll Converter

Convert Medium articles to Jekyll blog posts with proper formatting, front matter, and image handling.

## When to Use

Use this skill when the user provides a Medium URL and wants to:
- Convert it to Jekyll-compatible markdown
- Import a Medium post to their Jekyll blog
- Fetch and reformat a Medium article

## Process

### Step 1: Fetch the Medium Article

**Primary method**: Fetch using the web-reader tool:

```
mcp__web-reader__webReader with the provided Medium URL
```

**Fallback - RSS Feed**: If direct access is blocked (Cloudflare, etc.), use RSS feed:

```
curl -s -L "https://medium.com/feed/@username" > /tmp/medium_rss.xml
```

Then extract the specific article content from the RSS XML. RSS feed contains full article HTML in `<content:encoded>` tags.

### Step 2: Extract Metadata

From the fetched content, extract:
- **Title**: Article title
- **Published Date**: Original publication date
- **Tags/Categories**: Article tags if available

### Step 3: Generate Jekyll Front Matter

Create proper YAML front matter using this exact template:

```yaml
---
date: YYYY-MM-DD 00:00:00
layout: post
title: "Article Title"
type: [Guide|concept|experience|paper-review]
math: true

category: [Category]
tags:
  - Tag1
  - Tag2
author: pyy0715
---
```

**Date format**: Use `YYYY-MM-DD 00:00:00` (no timezone)

**type determination logic**:
- `Guide`: 가이드, 설정, 설치, 실행, 튜토리얼, How-to, 단계별 설명
- `concept`: 개요, 소개, 개념, 이론, 원리
- `experience`: 후기, 경험, 회고, 리뷰
- `paper-review`: 논문 리뷰, paper review

**category**: Single value from these options:
- `AI & ML`
- `DevOps`
- `Computer Science`
- `Software Engineering`
- `Career & Growth`

**tags**: Use YAML list format with each tag on a new line, indented with 2 spaces.

**author**: Always use `pyy0715`

### Step 4: Convert Content

#### Headings Processing

1. Preserve Original Headings

- Keep Medium `H1` as `H1`
- Keep Medium `H2` as `H2`
- Do not demote headings automatically

2. Structural Expansion (Generate H3 / H4 When Needed)

Medium is shallow (H1, H2 only).  
Expand hierarchy based on **semantic structure**, not formatting.

Promote to H3 when:

- A section contains clearly separable subtopics
- A process has distinct phases (Setup → Execute → Validate)
- A conceptual block introduces a new focus

Promote to H4 when:

- An H3 contains multiple meaningful internal divisions
- The content is long enough to require deeper scanning structure

3. Hierarchy Rules

- Maintain proper nesting: `H1 → H2 → H3 → H4`
- Do not skip levels
- Only expand if it improves clarity
- If a heading title feels vague, misleading, or structurally weak:
  - Propose a clearer alternative
  - Apply changes only after user approval

4. Do Not Expand

Do not create headings for:

- Short remarks
- Minor emphasis
- Decorative text
- Trivial lists

#### Images

**CRITICAL**: Follow these rules exactly for all images:

**1. URL Format Conversion**

RSS feeds provide URLs in this format:
```
https://cdn-images-1.medium.com/max/XXX/1*HASH.png
```

Convert ALL image URLs to miro format:
```
https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HASH.png
```

Conversion rule:
- Replace `cdn-images-1.medium.com/max/XXX/` with `miro.medium.com/v2/resize:fit:1400/format:webp/`
- Use `resize:fit:1400` as standard size (regardless of original size)

**2. Image Format**

```markdown
![descriptive alt text](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HASH.png) *caption text*
```

**3. Alt Text Rules (NEVER empty)**
- If original has alt text: use it
- If original has no alt text but has caption: derive alt text from caption
- If neither exists: create descriptive alt text from surrounding context
- Example: Caption "chronyc sources -v" → Alt text "chronyc sources"

**4. Caption Rules**
- Extract captions from `<figcaption>` tags in HTML/RSS
- If no figcaption: look for nearby descriptive text or command shown in screenshot
- Captions can contain markdown links: `*[link text](url)*`
- Place caption immediately after image URL with single space

**5. Complete Example**

From RSS:
```html
<figure><img alt="" src="https://cdn-images-1.medium.com/max/1024/1*abc123.png" /><figcaption>chronyc sources -v</figcaption></figure>
```

To Markdown:
```markdown
![chronyc sources](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*abc123.png) *chronyc sources -v*
```

**6. Images Without Captions**

If an image has no caption in the original:
- Create a descriptive caption based on what the image shows
- Example: Screenshot of terminal output → use the command as caption

#### Code Blocks

Convert to fenced code blocks with language hints. **IMPORTANT**: If code contains Liquid syntax ( `{%`, `{{` ), wrap with raw tags:

```markdown
{% raw %}
```yaml
code: here
with: liquid_syntax
```
{% endraw %}
```

Language hints to use:
- `yaml` for YAML/Ansible
- `bash` or `shell` for commands
- `python` for Python code
- `ruby` for Ruby code
- `json` for JSON
- `ini` for INI files
- `docker` for Dockerfile

#### Other Formatting

- **Bold/Italic**: Preserve `**bold**` and `*italic*`
- **Links**: Convert to `[text](url)` format
- **Lists**: Preserve ordered and unordered lists
- **Blockquotes**: Convert to `>` markdown quotes
- **Tables**: Convert to markdown table format if present

### Step 5: Generate Filename

Create filename in Jekyll format:
```
YYYY-MM-DD-title-slug.md
```

- Use the published date for YYYY-MM-DD
- Create URL-friendly slug from title (lowercase, hyphens for spaces, remove special characters)
- Example: `2024-03-15-kubernetes-ha-setup.md`

### Step 6: Save to Jekyll _posts

Save the converted markdown file to:
```
_posts/YYYY-MM-DD-title-slug.md
```

## Output Format

After conversion, provide:
1. The generated filename
2. A summary of the conversion (title, date, type, category, tags)
3. Any issues encountered (missing images, formatting problems)
4. Preview of front matter

## Example Usage

**User input:**
```
Medium URL: https://medium.com/@user/my-article-abc123
Category: DevOps
```

**Expected output:**
1. Fetch article content
2. Determine type based on content analysis
3. Generate front matter with proper format
4. Convert body content with heading demotion
5. Format images with inline captions
6. Wrap code blocks with raw tags if needed
7. Save to `_posts/2024-03-15-my-article.md`
8. Report results

## Handling Edge Cases

### Missing Publication Date
If no date is found, ask the user for the date or use today's date.

### Category Not Provided
Ask the user to specify category from: `AI & ML`, `DevOps`, `Computer Science`, `Software Engineering`, `Career & Growth`

### Complex Images
For images with complex captions or figures:
- Preserve all caption text
- Keep original image URLs
- Note any images that might need manual review

### Code Snippets Without Language
Default to no language hint if the language is unclear:
```markdown
```
code here
```
```

### Gists and Embedded Content
For GitHub Gists or embedded content:
- Note them in the output
- Suggest manual review for complex embeds

## Quality Checklist

Before saving, verify:

**Front Matter:**
- [ ] Date format is `YYYY-MM-DD 00:00:00` (no timezone)
- [ ] `type` field is present and valid (`Guide|concept|experience|paper-review`)
- [ ] `math: true` is included
- [ ] `category` is a single value, not array
- [ ] `tags` is in YAML list format (each tag on new line with 2-space indent)
- [ ] `author` is `pyy0715`

**Content Structure:**
- [ ] No H1 in content (only H2 and below)
- [ ] Headings are properly demoted (original H1→H2, H2→H3)

**Images (CRITICAL - verify ALL images):**
- [ ] ALL image URLs use `miro.medium.com/v2/resize:fit:1400/format:webp/` format
- [ ] NO images use `cdn-images-1.medium.com` URLs
- [ ] ALL images have alt text (never empty `![]`)
- [ ] ALL images have captions on same line: `![alt](url) *caption*`
- [ ] Captions are derived from `<figcaption>` or context if not provided

**Code Blocks:**
- [ ] Code blocks have appropriate language hints
- [ ] Code with Liquid syntax (`{%`, `{{`) wrapped in `{% raw %}` tags

---

## Review Workflow (팀 기반 검토)

For comprehensive quality review, use a 2-person team workflow:

### Team Setup

When the user requests a reviewed conversion, create a team:

```
TeamCreate with team_name="medium-converter"
```

**Team Roles:**
- **team-lead**: Performs Medium article conversion, collects user input, saves final post
- **reviewer**: Reviews Korean expressions, technical terms, structure using humanizer skill

### Workflow Steps

1. **User Input Collection**
   - Ask user for Medium URL and Category
   - Categories: `AI & ML | DevOps | Computer Science | Software Engineering | Career & Growth`

2. **Conversion (team-lead)**
   - Fetch article from Medium URL
   - Extract metadata and determine `type`
   - Convert content with proper formatting
   - Generate front matter
   - Send converted content to reviewer

3. **Quality Review (reviewer)**
   - Receive converted content from team-lead
   - Invoke humanizer skill: `Skill with skill="humanizer"` on the content
   - Check for:
     - AI-generated writing patterns
     - Unnatural Korean expressions
     - Technical term accuracy
     - Structure and flow
   - Send feedback to team-lead

4. **Finalization (team-lead)**
   - Apply reviewer feedback
   - Perform quality checklist
   - Save to `_posts/YYYY-MM-DD-slug.md`
   - Report completion

### Message Example (reviewer to team-lead)

```
SendMessage with:
  type: "message"
  recipient: "team-lead"
  content: |
    ## Review Feedback

    ### AI Patterns Found:
    - [Pattern description and fix]

    ### Korean Expression Issues:
    - [Issue and correction]

    ### Technical Terms:
    - [Term corrections]

    ### Structure:
    - [Suggestions]

    Apply these changes before saving.
  summary: "Review complete with feedback"
```

### Cleanup

After successful conversion and review:
```
SendMessage with type="shutdown_request" to teammates
TeamDelete after all teammates shut down
```
