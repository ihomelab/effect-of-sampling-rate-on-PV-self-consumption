import fetch from "node-fetch";
import { pipeline as _streamPipeline } from "stream";
import { promisify } from "util";
import fs from "fs";
import path from "path";
import ProgressBar from "progress";

const streamPipeline = promisify(_streamPipeline);

function twoDigit(n: number): string {
  return (n + "").padStart(2, "0");
}

async function getDatasetUrl(
  month: number, // 2 digit month starting at 1
  year: number, // 2 digit year
): Promise<string> {
  const url = [
    "https://www.pangaea.de/advanced/search.php?facets=default&q=ftp%3A%2F%2Fftp.bsrn.awi.de%2Fcam%2Fcam",
    twoDigit(month),
    twoDigit(year),
    ".dat.gz",
  ].join("");
  const response = await (await fetch(url)).json();
  if (response.results.length !== 1) {
    throw new Error("expected exactly 1 result");
  }
  const uri: string = response.results[0].URI;

  const prefix = "doi:";
  if (!uri.startsWith(prefix)) {
    throw new Error("invalid URI format");
  }
  return [
    "https://doi.pangaea.de/",
    uri.slice(prefix.length),
    "?format=textfile&charset=UTF-8",
  ].join("");
}

async function downloadDataset(
  url: string,
  destination: string,
  loginId: string,
  sessionId: string,
): Promise<void> {
  const res = await fetch(url, {
    headers: {
      accept:
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
      cookie: `PanLoginID=${loginId}; pansessid=${sessionId}`,
    },
  });
  if (!res.ok) {
    throw new Error(`request failed (${res.status})`);
  }
  return streamPipeline(res.body, fs.createWriteStream(destination));
}

async function main(): Promise<void> {
  if (process.argv.length !== 4) {
    console.error("Usage: node index.js <PanLoginID> <pansessid>");
    process.exit(1);
  }
  const loginId = process.argv[2];
  const sessionId = process.argv[3];

  const baseDir = path.join(__dirname, "../../data/bsrn");
  fs.mkdirSync(baseDir, { recursive: true });

  const tasks: [number, number][] = [];
  for (let year = 13; year <= 15; year++) {
    for (let month = 1; month <= 12; month++) {
      tasks.push([month, year]);
    }
  }

  const bar = new ProgressBar(":bar :percent", { total: tasks.length });
  for (const [month, year] of tasks) {
    const sourceUrl = await getDatasetUrl(month, year);
    const targetPath = path.join(
      baseDir,
      `cam${twoDigit(month)}${twoDigit(year)}.csv`,
    );
    await downloadDataset(sourceUrl, targetPath, loginId, sessionId);
    bar.tick();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
