#!/usr/bin/env python3
"""
Taxonomy Migration Script
Migrates blog posts to new category/tag structure
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
POSTS_DIR = Path("_posts")
MAPPING_FILE = Path("scripts/taxonomy_mapping.yaml")
DRY_RUN = False  # Set to True for preview mode


def load_mapping() -> Dict:
    """Load taxonomy mapping configuration"""
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_front_matter(content: str) -> tuple[Dict, str]:
    """
    Parse YAML front matter from markdown file
    Returns: (front_matter_dict, body_content)
    """
    # Match front matter between --- delimiters
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        raise ValueError("No valid front matter found")

    front_matter_str = match.group(1)
    body = match.group(2)

    # Parse YAML
    front_matter = yaml.safe_load(front_matter_str)

    return front_matter, body


def migrate_front_matter(front_matter: Dict, mapping: Dict, filename: str) -> Dict:
    """
    Migrate front matter to new taxonomy structure
    """
    migrated = front_matter.copy()

    # 1. Migrate Category
    old_category = front_matter.get('category', '')
    if old_category in mapping['category_mapping']:
        migrated['category'] = mapping['category_mapping'][old_category]
        print(f"  Category: {old_category} → {migrated['category']}")

    # 2. Migrate Subtitle → Type
    old_subtitle = front_matter.get('subtitle', '')

    # Check post-specific overrides first
    if filename in mapping.get('post_overrides', {}):
        override = mapping['post_overrides'][filename]
        if 'type' in override:
            migrated['type'] = override['type']
            print(f"  Type: (override) → {migrated['type']}")
    elif old_subtitle in mapping['type_mapping']:
        migrated['type'] = mapping['type_mapping'][old_subtitle]
        print(f"  Type: {old_subtitle} → {migrated['type']}")
    else:
        # Default to 'concept' if no subtitle
        migrated['type'] = 'concept'
        print(f"  Type: (empty) → concept")

    # Remove old subtitle field
    if 'subtitle' in migrated:
        del migrated['subtitle']

    # 3. Standardize Tags
    if 'tags' in front_matter and front_matter['tags']:
        old_tags = front_matter['tags']
        new_tags = []

        for tag in old_tags:
            # Apply tag replacements
            if tag in mapping.get('tag_replacements', {}):
                new_tag = mapping['tag_replacements'][tag]
                new_tags.append(new_tag)
                print(f"  Tag: {tag} → {new_tag}")
            else:
                new_tags.append(tag)

        # Remove duplicate tags
        new_tags = list(dict.fromkeys(new_tags))

        # Remove tags that are same as category (reduce redundancy)
        category = migrated.get('category', '')
        new_tags = [t for t in new_tags if t != category]

        migrated['tags'] = new_tags

    return migrated


def serialize_front_matter(front_matter: Dict) -> str:
    """
    Serialize front matter dict back to YAML string
    Maintains specific field order
    """
    # Define desired field order
    field_order = [
        'date', 'layout', 'title', 'type', 'math', 'image',
        'optimized_image', 'category', 'tags', 'author'
    ]

    lines = ['---']

    # Add fields in specific order
    for field in field_order:
        if field in front_matter:
            value = front_matter[field]

            if field == 'tags' and isinstance(value, list):
                # Format tags as YAML list
                lines.append('tags:')
                for tag in value:
                    lines.append(f'  - {tag}')
            elif isinstance(value, str):
                # String values (handle multiline if needed)
                if '\n' in value or ':' in value:
                    lines.append(f'{field}: |')
                    for line in value.split('\n'):
                        lines.append(f'  {line}')
                else:
                    lines.append(f'{field}: {value}')
            elif isinstance(value, bool):
                lines.append(f'{field}: {str(value).lower()}')
            else:
                lines.append(f'{field}: {value}')

    # Add any remaining fields not in field_order
    for key, value in front_matter.items():
        if key not in field_order:
            if isinstance(value, str):
                lines.append(f'{key}: {value}')
            else:
                lines.append(f'{key}: {value}')

    lines.append('---')
    return '\n'.join(lines)


def migrate_post(post_path: Path, mapping: Dict, dry_run: bool = False) -> bool:
    """
    Migrate a single post file
    Returns True if changes were made
    """
    filename = post_path.name
    print(f"\n📝 Processing: {filename}")

    try:
        # Read file
        with open(post_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse front matter
        front_matter, body = parse_front_matter(content)

        # Migrate front matter
        migrated_fm = migrate_front_matter(front_matter, mapping, filename)

        # Check if changes were made
        if migrated_fm == front_matter:
            print("  ✓ No changes needed")
            return False

        # Serialize back to YAML
        new_front_matter_str = serialize_front_matter(migrated_fm)
        new_content = f"{new_front_matter_str}\n{body}"

        # Write back (if not dry run)
        if not dry_run:
            with open(post_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  ✅ Migrated successfully")
        else:
            print("  🔍 DRY RUN - Changes not written")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Main migration function"""
    print("🚀 Starting Taxonomy Migration\n")
    print(f"Posts directory: {POSTS_DIR}")
    print(f"Mapping file: {MAPPING_FILE}")
    print(f"Dry run: {DRY_RUN}\n")

    # Load mapping
    try:
        mapping = load_mapping()
        print("✓ Loaded mapping configuration\n")
    except Exception as e:
        print(f"❌ Failed to load mapping: {e}")
        return

    # Get all post files
    post_files = sorted(POSTS_DIR.glob("*.md"))
    print(f"Found {len(post_files)} posts\n")
    print("=" * 60)

    # Migrate each post
    migrated_count = 0
    for post_path in post_files:
        if migrate_post(post_path, mapping, DRY_RUN):
            migrated_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"\n📊 Summary:")
    print(f"  Total posts: {len(post_files)}")
    print(f"  Migrated: {migrated_count}")
    print(f"  Unchanged: {len(post_files) - migrated_count}")

    if DRY_RUN:
        print("\n⚠️  DRY RUN MODE - No files were modified")
        print("   Set DRY_RUN = False to apply changes")
    else:
        print("\n✅ Migration complete!")


if __name__ == "__main__":
    main()
