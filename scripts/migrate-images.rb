#!/usr/bin/env ruby
require 'net/http'
require 'uri'
require 'fileutils'
require 'digest'

IMAGE_DIR = 'assets/img/posts'
FileUtils.mkdir_p(IMAGE_DIR)

puts "Starting image migration..."
puts "Image directory: #{IMAGE_DIR}"
puts

# Track statistics
stats = {
  total_images: 0,
  downloaded: 0,
  already_exist: 0,
  failed: 0,
  updated_posts: 0
}

# Find all posts
post_files = Dir.glob('_posts/*.md').sort
puts "Found #{post_files.length} posts to scan"
puts

post_files.each do |post_file|
  content = File.read(post_file)
  modified = false
  post_basename = File.basename(post_file, '.md')

  # Find all GitHub user-images URLs
  github_images = content.scan(/!\[([^\]]*)\]\((https:\/\/user-images\.githubusercontent\.com\/[^\)]+)\)/)

  if github_images.any?
    puts "📄 #{File.basename(post_file)} - found #{github_images.length} GitHub images"

    github_images.each_with_index do |(alt_text, url), index|
      stats[:total_images] += 1

      # Generate filename using post name and URL hash
      url_hash = Digest::MD5.hexdigest(url)[0..7]
      extension = File.extname(URI.parse(url).path)
      extension = '.png' if extension.empty?
      filename = "#{post_basename}-#{url_hash}#{extension}"
      local_path = "/#{IMAGE_DIR}/#{filename}"
      full_path = "#{IMAGE_DIR}/#{filename}"

      # Check if image already exists
      if File.exist?(full_path)
        puts "  ✓ Already exists: #{filename}"
        stats[:already_exist] += 1
      else
        # Download image
        begin
          uri = URI.parse(url)
          http = Net::HTTP.new(uri.host, uri.port)
          http.use_ssl = true
          http.read_timeout = 30

          response = http.get(uri.request_uri)

          if response.is_a?(Net::HTTPSuccess)
            File.binwrite(full_path, response.body)
            file_size = (response.body.bytesize / 1024.0).round(2)
            puts "  ⬇️  Downloaded: #{filename} (#{file_size} KB)"
            stats[:downloaded] += 1
          else
            puts "  ✗ Failed: #{url} (HTTP #{response.code})"
            stats[:failed] += 1
            next
          end
        rescue => e
          puts "  ✗ Error: #{url} - #{e.message}"
          stats[:failed] += 1
          next
        end
      end

      # Replace URL in content
      content.gsub!(url, local_path)
      modified = true
    end

    # Save updated post file
    if modified
      File.write(post_file, content)
      stats[:updated_posts] += 1
      puts "  ✓ Updated post file"
    end
    puts
  end
end

# Print summary
puts "=" * 60
puts "Migration Summary:"
puts "=" * 60
puts "Total images found:    #{stats[:total_images]}"
puts "Downloaded:            #{stats[:downloaded]}"
puts "Already existed:       #{stats[:already_exist]}"
puts "Failed:                #{stats[:failed]}"
puts "Posts updated:         #{stats[:updated_posts]}"
puts "=" * 60
puts
puts "✓ Image migration complete!"
