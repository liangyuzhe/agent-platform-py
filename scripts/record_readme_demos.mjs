import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const baseUrl = process.env.DEMO_BASE_URL || "http://localhost:8080/#/";
const outDir = path.resolve("docs/assets/demos");
const rawDir = path.join(outDir, "raw");
const gifTrimStart = Number(process.env.DEMO_GIF_TRIM_START || "3.2");

const demos = [
  {
    slug: "chat-maotai-q1-profit",
    title: "Chat / Knowledge: 茅台第一季度盈利",
    questions: ["茅台第一季度盈利"],
    approvals: 0,
  },
  {
    slug: "sql-last-year-profit-approved",
    title: "SQL Query: 公司盈利",
    questions: ["公司盈利"],
    approvals: 2,
    waitForReflection: false,
  },
  {
    slug: "sql-loss-followup-repair-approved",
    title: "SQL Repair: 执行失败 -> 反思修正 -> 结果",
    questions: ["2024年每月亏损金额"],
    approvals: 5,
    waitForReflection: true,
  },
];

const filter = process.env.DEMO_FILTER;
const selected = filter ? demos.filter((demo) => demo.slug.includes(filter)) : demos;

function ensureDirs() {
  fs.mkdirSync(outDir, { recursive: true });
  fs.mkdirSync(rawDir, { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function addDemoOverlay(page, title) {
  await page.addStyleTag({
    content: `
      .readme-demo-overlay {
        position: fixed;
        top: 14px;
        right: 18px;
        z-index: 99999;
        padding: 9px 12px;
        border-radius: 8px;
        background: rgba(16, 24, 32, .88);
        color: #fff;
        font: 600 13px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        box-shadow: 0 10px 28px rgba(0,0,0,.22);
      }
    `,
  });
  await page.evaluate((text) => {
    const existing = document.querySelector(".readme-demo-overlay");
    if (existing) existing.remove();
    const div = document.createElement("div");
    div.className = "readme-demo-overlay";
    div.textContent = text;
    document.body.appendChild(div);
  }, title);
}

async function waitForAgentIdle(page) {
  await page.waitForFunction(() => {
    const btn = document.querySelector("#agentBtn");
    return btn && !btn.disabled;
  }, undefined, { timeout: 180_000 });
}

async function submitQuestion(page, question) {
  const input = page.locator("#sqlQuery");
  await input.click();
  await input.fill("");
  await input.pressSequentially(question, { delay: 45 });
  await sleep(350);
  await page.locator("#agentBtn").click();
  await waitForAgentIdle(page);
  await sleep(1_000);
}

async function approveVisibleSql(page, maxApprovals) {
  for (let i = 0; i < maxApprovals; i += 1) {
    const approve = page.locator(".btn-approve:not([disabled])").last();
    try {
      await approve.waitFor({ state: "visible", timeout: 12_000 });
    } catch {
      break;
    }
    const assistantCount = await page.locator("#sqlMessages .sql-message.assistant .bubble").count();
    await sleep(800);
    await approve.click();
    await waitForApproveSettled(page, assistantCount);
    await sleep(2_500);
  }
}

async function waitForApproveSettled(page, previousAssistantCount) {
  await page.waitForFunction((before) => {
    const bubbles = Array.from(document.querySelectorAll("#sqlMessages .sql-message.assistant .bubble"));
    if (bubbles.length <= before) return false;
    const last = bubbles[bubbles.length - 1];
    const text = last.innerText || "";
    const hasFinalResult = Boolean(last.querySelector(".intent-tag.result"))
      || text.includes("查询已执行完成")
      || text.includes("共返回")
      || text.includes("SQL 执行完成");
    const hasNextApproval = Boolean(last.querySelector(".btn-approve:not([disabled])"));
    const hasTerminalError = text.includes("系统错误") || text.includes("SQL 执行失败");
    return hasFinalResult || hasNextApproval || hasTerminalError;
  }, previousAssistantCount, { timeout: 180_000 });
}

async function waitForFinalSqlResult(page) {
  await page.waitForFunction(() => {
    const bubbles = Array.from(document.querySelectorAll("#sqlMessages .sql-message.assistant .bubble"));
    return bubbles.some((bubble) => {
      const text = bubble.innerText || "";
      return Boolean(bubble.querySelector(".intent-tag.result"))
        && (text.includes("查询已执行完成") || text.includes("共返回"));
    });
  }, undefined, { timeout: 180_000 });
}

async function waitForReflectionFlowVisible(page) {
  await page.waitForFunction(() => {
    const text = document.querySelector("#sqlMessages")?.innerText || "";
    return text.includes("检测到执行失败或结果异常")
      && text.includes("请确认是否执行修正后的 SQL")
      && text.includes("查询已执行完成");
  }, undefined, { timeout: 240_000 });
}

function convertToGif(webmPath, gifPath) {
  const palette = path.join(rawDir, `${path.basename(gifPath, ".gif")}-palette.png`);
  execFileSync("ffmpeg", [
    "-y",
    "-ss", String(gifTrimStart),
    "-i", webmPath,
    "-vf", "fps=8,scale=960:-1:flags=lanczos,palettegen",
    palette,
  ], { stdio: "inherit" });
  execFileSync("ffmpeg", [
    "-y",
    "-ss", String(gifTrimStart),
    "-i", webmPath,
    "-i", palette,
    "-filter_complex", "fps=8,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
    gifPath,
  ], { stdio: "inherit" });
}

async function recordDemo(browser, demo) {
  console.log(`Recording ${demo.slug}: ${demo.title}`);
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: { dir: rawDir, size: { width: 1280, height: 720 } },
  });
  const page = await context.newPage();
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator("button", { hasText: "SQL Agent" }).click();
  await addDemoOverlay(page, demo.title);
  await sleep(900);

  for (const question of demo.questions) {
    await submitQuestion(page, question);
    await approveVisibleSql(page, demo.approvals);
    await sleep(900);
  }

  if (demo.approvals > 0) {
    await waitForFinalSqlResult(page);
  }
  if (demo.waitForReflection) {
    await waitForReflectionFlowVisible(page);
  }
  await sleep(5_000);
  const video = page.video();
  await context.close();
  const recorded = await video.path();
  const rawPath = path.join(rawDir, `${demo.slug}.webm`);
  fs.renameSync(recorded, rawPath);
  const gifPath = path.join(outDir, `${demo.slug}.gif`);
  convertToGif(rawPath, gifPath);
  console.log(`Wrote ${gifPath}`);
}

async function main() {
  ensureDirs();
  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--window-size=1280,720"],
  });
  try {
    for (const demo of selected) {
      await recordDemo(browser, demo);
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
