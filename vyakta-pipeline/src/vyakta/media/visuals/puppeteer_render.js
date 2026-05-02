const puppeteer = require('puppeteer');
const fs = require('fs');

async function render(configPath) {
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({
      width: config.width,
      height: config.height,
      deviceScaleFactor: 1,
    });

    for (const slide of config.slides) {
      const html = fs.readFileSync(slide.html_path, 'utf8');
      await page.setContent(html, { waitUntil: 'networkidle0' });
      await page.screenshot({
        path: slide.output_path,
        type: 'png',
        fullPage: false,
      });
    }
  } finally {
    await browser.close();
  }
}

const configPath = process.argv[2];
if (!configPath) {
  console.error('Usage: node puppeteer_render.js <config.json>');
  process.exit(1);
}

render(configPath).catch((err) => {
  console.error(err);
  process.exit(1);
});
