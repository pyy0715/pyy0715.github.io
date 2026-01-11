/**
 * 블로그 이미지 자동 최적화 스크립트
 * 
 * 기능:
 * - PNG/JPG → WebP 변환 (60%+ 용량 절감)
 * - 썸네일 자동 생성 (커버 이미지용)
 * - 이미 최적화된 이미지는 스킵
 * 
 * 사용법:
 *   npm install sharp glob
 *   node scripts/optimize-images.js          # 전체 최적화
 *   node scripts/optimize-images.js --dry    # 미리보기 (실행 안함)
 *   node scripts/optimize-images.js --force  # 기존 파일도 덮어쓰기
 */

const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// ============================================
// 설정 (필요에 따라 수정)
// ============================================
const CONFIG = {
  // 이미지 소스 디렉토리
  srcDir: 'assets/img/posts',
  
  // WebP 품질 (0-100, 높을수록 고품질/큰용량)
  quality: {
    full: 80,      // 본문 이미지
    thumb: 70      // 썸네일
  },
  
  // 썸네일 너비 (px)
  thumbWidth: 400,
  
  // 본문 이미지 최대 너비 (이보다 크면 리사이즈)
  maxWidth: 1200,
  
  // 지원 확장자
  extensions: ['.png', '.jpg', '.jpeg']
};

// ============================================
// 유틸리티 함수
// ============================================
function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function getImageFiles(dir) {
  if (!fs.existsSync(dir)) {
    console.error(`❌ 디렉토리가 존재하지 않습니다: ${dir}`);
    process.exit(1);
  }
  
  return fs.readdirSync(dir)
    .filter(file => CONFIG.extensions.includes(path.extname(file).toLowerCase()))
    .map(file => path.join(dir, file));
}

// ============================================
// 이미지 최적화
// ============================================
async function optimizeImage(inputPath, options = {}) {
  const { dryRun = false, force = false } = options;
  const filename = path.basename(inputPath);
  const dir = path.dirname(inputPath);
  const nameWithoutExt = filename.replace(/\.(png|jpg|jpeg)$/i, '');
  
  const webpPath = path.join(dir, `${nameWithoutExt}.webp`);
  const thumbPath = path.join(dir, `${nameWithoutExt}-thumb.webp`);
  
  // 이미 최적화된 경우 스킵
  if (!force && fs.existsSync(webpPath) && fs.existsSync(thumbPath)) {
    return { skipped: true, filename };
  }
  
  const originalStats = fs.statSync(inputPath);
  const originalSize = originalStats.size;
  
  if (dryRun) {
    console.log(`[DRY] ${filename} → ${nameWithoutExt}.webp, ${nameWithoutExt}-thumb.webp`);
    return { dryRun: true, filename };
  }
  
  try {
    const metadata = await sharp(inputPath).metadata();
    
    // 1. WebP 변환 (필요시 리사이즈)
    let pipeline = sharp(inputPath);
    if (metadata.width > CONFIG.maxWidth) {
      pipeline = pipeline.resize(CONFIG.maxWidth, null, { withoutEnlargement: true });
    }
    await pipeline.webp({ quality: CONFIG.quality.full }).toFile(webpPath);
    
    // 2. 썸네일 생성
    await sharp(inputPath)
      .resize(CONFIG.thumbWidth, null, { withoutEnlargement: true })
      .webp({ quality: CONFIG.quality.thumb })
      .toFile(thumbPath);
    
    const webpStats = fs.statSync(webpPath);
    const thumbStats = fs.statSync(thumbPath);
    const savedPercent = ((1 - webpStats.size / originalSize) * 100).toFixed(0);
    
    return {
      success: true,
      filename,
      originalSize,
      webpSize: webpStats.size,
      thumbSize: thumbStats.size,
      savedPercent
    };
    
  } catch (error) {
    return { error: true, filename, message: error.message };
  }
}

// ============================================
// 메인 실행
// ============================================
async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry');
  const force = args.includes('--force');
  
  console.log('');
  console.log('🖼️  블로그 이미지 최적화');
  console.log('========================');
  if (dryRun) console.log('⚠️  DRY RUN 모드 (실제 변환 안함)');
  if (force) console.log('⚠️  FORCE 모드 (기존 파일 덮어쓰기)');
  console.log('');
  
  const images = getImageFiles(CONFIG.srcDir);
  console.log(`📁 ${CONFIG.srcDir}`);
  console.log(`📷 ${images.length}개 이미지 발견\n`);
  
  if (images.length === 0) {
    console.log('처리할 이미지가 없습니다.');
    return;
  }
  
  // 통계
  let stats = {
    processed: 0,
    skipped: 0,
    errors: 0,
    originalTotal: 0,
    webpTotal: 0,
    thumbTotal: 0
  };
  
  // 처리
  for (const imagePath of images) {
    const result = await optimizeImage(imagePath, { dryRun, force });
    
    if (result.skipped) {
      stats.skipped++;
      process.stdout.write('.');
    } else if (result.dryRun) {
      stats.processed++;
    } else if (result.error) {
      stats.errors++;
      console.log(`\n❌ ${result.filename}: ${result.message}`);
    } else if (result.success) {
      stats.processed++;
      stats.originalTotal += result.originalSize;
      stats.webpTotal += result.webpSize;
      stats.thumbTotal += result.thumbSize;
      console.log(`✅ ${result.filename} → WebP (${result.savedPercent}% ↓)`);
    }
  }
  
  // 결과 요약
  console.log('\n');
  console.log('========================');
  console.log('📊 결과 요약');
  console.log(`   처리됨: ${stats.processed}`);
  console.log(`   스킵됨: ${stats.skipped} (이미 최적화됨)`);
  console.log(`   에러: ${stats.errors}`);
  
  if (stats.originalTotal > 0) {
    const totalSaved = ((1 - stats.webpTotal / stats.originalTotal) * 100).toFixed(1);
    console.log('');
    console.log(`   원본 총합: ${formatSize(stats.originalTotal)}`);
    console.log(`   WebP 총합: ${formatSize(stats.webpTotal)} (${totalSaved}% 절감)`);
    console.log(`   썸네일 총합: ${formatSize(stats.thumbTotal)}`);
  }
  console.log('');
}

main().catch(console.error);
