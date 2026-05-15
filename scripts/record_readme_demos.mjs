import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const baseUrl = process.env.DEMO_BASE_URL || "http://localhost:8080/#/";
const outDir = path.resolve("docs/assets/demos");
const rawDir = path.join(outDir, "raw");
const gifTrimStart = Number(process.env.DEMO_GIF_TRIM_START || "3.2");
const gifSpeed = Number(process.env.DEMO_GIF_SPEED || "3");
const browserChannel = process.env.DEMO_BROWSER_CHANNEL || "chrome";
const require = createRequire(import.meta.url);

const demos = [
  {
    slug: "chat-maotai-q1-profit",
    title: "Chat / Knowledge: 贵州茅台第一季度盈利",
    questions: ["贵州茅台第一季度盈利"],
    approvals: 0,
    finalText: "茅台",
  },
  {
    slug: "sql-last-year-loss-approved",
    title: "SQL Query: 去年亏损",
    questions: ["去年亏损"],
    approvals: 3,
    finalText: "查询已执行完成",
  },
  {
    slug: "sql-loss-followup-amount-approved",
    title: "Multi-turn SQL: 亏损多少",
    questions: ["去年亏损", "亏损多少"],
    approvals: 3,
    finalText: "亏损",
  },
  {
    slug: "sql-q1-salary-approved",
    title: "SQL Query: 第一季度员工工资",
    questions: ["2026年第一季度按月统计应付职工薪酬借方和贷方金额"],
    approvals: 2,
    finalText: "查询已执行完成",
  },
  {
    slug: "sql-management-user-role-approved",
    title: "SQL Query: 用户真实姓名与角色",
    questions: ["查询所有用户的真实姓名以及他们被分配的角色名称"],
    approvals: 2,
    finalText: "查询已执行完成",
  },
  {
    slug: "sql-complex-finance-relation-plan-approved",
    title: "Complex Plan: 收入成本预算回款费用关系",
    questions: [
      "收入成本预算回款费用之间的关系",
    ],
    approvals: 1,
    finalText: "复杂查询计划执行完成",
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

async function loadChromium() {
  try {
    const mod = await import("playwright");
    return mod.chromium;
  } catch (error) {
    if (process.env.PLAYWRIGHT_MODULE) {
      return require(process.env.PLAYWRIGHT_MODULE).chromium;
    }
    throw error;
  }
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
      || text.includes("SQL 执行完成")
      || text.includes("复杂计划已确认")
      || text.includes("复杂查询计划执行完成")
      || text.includes("分步执行将在下一迭代启用");
    const hasNextApproval = Boolean(last.querySelector(".btn-approve:not([disabled])"));
    const hasTerminalError = text.includes("系统错误") || text.includes("SQL 执行失败");
    return hasFinalResult || hasNextApproval || hasTerminalError;
  }, previousAssistantCount, { timeout: 180_000 });
}

async function waitForFinalText(page, expectedText) {
  if (!expectedText) return;
  await page.waitForFunction((needle) => {
    const text = document.querySelector("#sqlMessages")?.innerText || "";
    return text.includes(needle);
  }, expectedText, { timeout: 180_000 });
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

async function focusFinalResult(page) {
  await page.evaluate(() => {
    const container = document.querySelector("#sqlMessages");
    if (!container) return;

    const assistantBubbles = Array.from(container.querySelectorAll(".sql-message.assistant .bubble"));
    const finalBubble = assistantBubbles.at(-1);
    if (!finalBubble) return;

    // Collapse earlier SQL detail blocks so the final result is actually visible in-frame.
    const openDetails = Array.from(container.querySelectorAll("details[open]"));
    for (const detail of openDetails) {
      if (!finalBubble.contains(detail)) {
        detail.removeAttribute("open");
      }
    }

    const finalText = finalBubble.innerText || "";
    if (finalText.includes("复杂查询计划执行完成")) {
      container.scrollTop = finalBubble.offsetTop + finalBubble.scrollHeight - container.clientHeight + 24;
    } else {
      container.scrollTop = container.scrollHeight;
      finalBubble.scrollIntoView({ block: "center", inline: "nearest" });
    }
  });
  await sleep(1200);
}

function convertToGif(webmPath, gifPath) {
  const palette = path.join(rawDir, `${path.basename(gifPath, ".gif")}-palette.png`);
  const speedFactor = gifSpeed > 0 ? (1 / gifSpeed).toFixed(4) : "1";
  const videoFilter = `setpts=${speedFactor}*PTS,fps=8,scale=960:-1:flags=lanczos`;
  execFileSync("ffmpeg", [
    "-y",
    "-ss", String(gifTrimStart),
    "-i", webmPath,
    "-vf", `${videoFilter},palettegen`,
    palette,
  ], { stdio: "inherit" });
  execFileSync("ffmpeg", [
    "-y",
    "-ss", String(gifTrimStart),
    "-i", webmPath,
    "-i", palette,
    "-filter_complex", `${videoFilter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5`,
    gifPath,
  ], { stdio: "inherit" });
}

async function recordDemo(browser, demo) {
  console.log(`Recording ${demo.slug}: ${demo.title}`);
  const demoSessionId = `readme_demo_${demo.slug}_${Date.now()}`;
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: { dir: rawDir, size: { width: 1280, height: 720 } },
  });
  await context.addInitScript((sessionId) => {
    window.__DEMO_SESSION_ID = sessionId;
  }, demoSessionId);
  const page = await context.newPage();
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await addDemoOverlay(page, demo.title);
  await sleep(900);

  if (demo.setupRouteRule) {
    await setupQueryRouteRule(page, demo.setupRouteRule);
  }

  await page.locator("button", { hasText: "SQL Agent" }).click();
  await sleep(700);

  for (const question of demo.questions) {
    await submitQuestion(page, question);
    await approveVisibleSql(page, demo.approvals);
    await sleep(900);
  }

  if (demo.approvals > 0 && demo.finalText?.includes("查询已执行完成")) {
    await waitForFinalSqlResult(page);
  }
  await waitForFinalText(page, demo.finalText);
  if (demo.waitForReflection) {
    await waitForReflectionFlowVisible(page);
  }
  await focusFinalResult(page);
  await sleep(6_000);
  const video = page.video();
  await context.close();
  const recorded = await video.path();
  const rawPath = path.join(rawDir, `${demo.slug}.webm`);
  fs.renameSync(recorded, rawPath);
  const gifPath = path.join(outDir, `${demo.slug}.gif`);
  convertToGif(rawPath, gifPath);
  console.log(`Wrote ${gifPath}`);
}

async function setupQueryRouteRule(page, rule) {
  await page.evaluate(() => switchTab("admin"));
  await page.waitForFunction(() => document.querySelector("#tab-admin.active") !== null, undefined, { timeout: 30_000 });
  await page.waitForSelector("#queryRouteRuleName", { state: "visible", timeout: 30_000 });
  await page.locator("#queryRouteRuleName").fill(rule.name);
  await page.locator("#queryRouteRuleSignal").selectOption(rule.signal);
  await page.locator("#queryRouteRuleMatchType").selectOption(rule.matchType);
  await page.locator("#queryRouteRulePattern").fill(rule.pattern);
  await page.locator("#queryRouteRulePriority").fill(rule.priority);
  await page.locator("#queryRouteRuleConfidence").fill(rule.confidence);
  await page.locator("#queryRouteRuleDescription").fill(rule.description);
  const enabled = page.locator("#queryRouteRuleEnabled");
  if (!(await enabled.isChecked())) {
    await enabled.check();
  }
  await sleep(500);
  await page.locator("button", { hasText: "保存规则" }).last().click();
  await page.waitForFunction(() => {
    const text = document.querySelector("#queryRouteRuleStatus")?.innerText || "";
    return text.includes("已加载") || text.includes("保存失败");
  }, undefined, { timeout: 30_000 });
  await sleep(900);
  await page.evaluate(() => switchTab("agent"));
  await page.waitForFunction(() => document.querySelector("#tab-agent.active") !== null, undefined, { timeout: 30_000 });
}

async function main() {
  ensureDirs();
  const chromium = await loadChromium();
  const launchOptions = {
    headless: true,
    args: ["--window-size=1280,720"],
  };
  if (browserChannel) {
    launchOptions.channel = browserChannel;
  }
  const browser = await chromium.launch(launchOptions);
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
