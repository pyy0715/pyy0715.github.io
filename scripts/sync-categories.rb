#!/usr/bin/env ruby
require 'yaml'
require 'set'
require 'date'

# 포스트에서 카테고리 수집
categories = Set.new
Dir.glob('_posts/*.md').each do |post_file|
  content = File.read(post_file)
  if content =~ /^---\s*\n(.*?)\n---/m
    frontmatter = YAML.safe_load($1, permitted_classes: [Time, Date, Symbol])
    categories.add(frontmatter['category']) if frontmatter['category']
  end
end

# 기존 카테고리 파일 확인
existing_categories = Dir.glob('category/*.md').map do |f|
  File.basename(f, '.md')
end

# 새 카테고리 파일 생성
categories.each do |category|
  slug = category.downcase.gsub(/\s+/, '-')
  file_path = "category/#{slug}.md"

  unless File.exist?(file_path)
    content = <<~CONTENT
      ---
      layout: category
      title: #{category}
      slug: #{category}
      description: Posts about #{category}
      ---
    CONTENT

    File.write(file_path, content)
    puts "✓ Created: #{file_path}"
  end
end

# 고아 카테고리 찾기
orphaned = existing_categories - categories.map { |c| c.downcase.gsub(/\s+/, '-') }
if orphaned.any?
  puts "\n⚠️  Orphaned categories (no posts):"
  orphaned.each { |o| puts "  - category/#{o}.md" }
  puts "\nConsider removing these files."
end

puts "\n✓ Category sync complete!"
