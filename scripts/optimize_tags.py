#!/usr/bin/env python3
"""
Tags Optimization Script
Optimizes tags for all blog posts based on predefined mapping
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List

# Configuration
POSTS_DIR = Path("_posts")
MAPPING_FILE = Path("scripts/tags_optimization.yaml")
DRY_RUN = False  # Set to True for preview mode

def load_mapping() -> Dict:
    """Load tags optimization mapping"""
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_front_matter(content: str) -> tuple[Dict, str]:
    """Parse YAML front matter from markdown file"""
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        raise ValueError("No valid front matter found")

    front_matter_str = match.group(1)
    body = match.group(2)
    front_matter = yaml.safe_load(front_matter_str)

    return front_matter, body


def optimize_tags(front_matter: Dict, new_tags: List[str], filename: str) -> Dict:
    """Optimize tags for a post"""
    optimized = front_matter.copy()
    old_tags = front_matter.get('tags', [])

    # Update tags
    optimized['tags'] = new_tags

    # Print changes
    print(f"  Old tags ({len(old_tags)}): {', '.join(old_tags)}")
    print(f"  New tags ({len(new_tags)}): {', '.join(new_tags)}")

    # Show differences
    added = set(new_tags) - set(old_tags)
    removed = set(old_tags) - set(new_tags)

    if added:
        print(f"  ➕ Added: {', '.join(sorted(added))}")
    if removed:
        print(f"  ➖ Removed: {', '.join(sorted(removed))}")

    return optimized


def serialize_front_matter(front_matter: Dict) -> str:
    """Serialize front matter dict back to YAML string"""
    field_order = [
        'date', 'layout', 'title', 'type', 'math', 'image',
        'optimized_image', 'category', 'tags', 'author'
    ]

    lines = ['---']

    for field in field_order:
        if field in front_matter:
            value = front_matter[field]

            if field == 'tags' and isinstance(value, list):
                lines.append('tags:')
                for tag in value:
                    lines.append(f'  - {tag}')
            elif isinstance(value, str):
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

    # Add remaining fields
    for key, value in front_matter.items():
        if key not in field_order:
            if isinstance(value, str):
                lines.append(f'{key}: {value}')
            else:
                lines.append(f'{key}: {value}')

    lines.append('---')
    return '\n'.join(lines)


def optimize_post(post_path: Path, new_tags: List[str], dry_run: bool = False) -> bool:
    """Optimize tags for a single post"""
    filename = post_path.name
    print(f"\n📝 Processing: {filename}")

    try:
        # Read file
        with open(post_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse front matter
        front_matter, body = parse_front_matter(content)

        # Optimize tags
        optimized_fm = optimize_tags(front_matter, new_tags, filename)

        # Check if changes were made
        if optimized_fm.get('tags') == front_matter.get('tags'):
            print("  ✓ No changes needed")
            return False

        # Serialize back to YAML
        new_front_matter_str = serialize_front_matter(optimized_fm)
        new_content = f"{new_front_matter_str}\n{body}"

        # Write back (if not dry run)
        if not dry_run:
            with open(post_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  ✅ Optimized successfully")
        else:
            print("  🔍 DRY RUN - Changes not written")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Main optimization function"""
    print("🏷️  Starting Tags Optimization\n")
    print(f"Posts directory: {POSTS_DIR}")
    print(f"Mapping file: {MAPPING_FILE}")
    print(f"Dry run: {DRY_RUN}\n")

    # Load mapping
    try:
        mapping = load_mapping()
        print(f"✓ Loaded mapping for {len(mapping)} posts\n")
    except Exception as e:
        print(f"❌ Failed to load mapping: {e}")
        return

    print("=" * 60)

    # Optimize each post
    optimized_count = 0
    skipped_count = 0

    for filename, new_tags in mapping.items():
        post_path = POSTS_DIR / filename

        if not post_path.exists():
            print(f"\n⚠️  Post not found: {filename}")
            skipped_count += 1
            continue

        if optimize_post(post_path, new_tags, DRY_RUN):
            optimized_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"\n📊 Summary:")
    print(f"  Total in mapping: {len(mapping)}")
    print(f"  Optimized: {optimized_count}")
    print(f"  Skipped: {skipped_count}")

    if DRY_RUN:
        print("\n⚠️  DRY RUN MODE - No files were modified")
        print("   Set DRY_RUN = False to apply changes")
    else:
        print("\n✅ Optimization complete!")

    # Statistics
    print(f"\n📈 Tag Statistics:")
    total_tags = sum(len(tags) for tags in mapping.values())
    avg_tags = total_tags / len(mapping) if mapping else 0
    print(f"  Average tags per post: {avg_tags:.1f}")
    print(f"  Recommended range: 3-5 tags")


if __name__ == "__main__":
    main()
