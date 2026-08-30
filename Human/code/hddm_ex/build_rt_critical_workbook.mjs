import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const baseDir = process.env.RT_CRITICAL_BASE_DIR
  ?? scriptDir;
const outputDir = path.join(baseDir, "outputs", "rt_critical_complete_data_20260806");
const csvDir = path.join(outputDir, "csv");
const qaDir = path.join(baseDir, "work", "rt_critical_workbook_qa_20260806");

const summary = JSON.parse(
  await fs.readFile(path.join(outputDir, "package_summary.json"), "utf8"),
);

const imports = [
  {
    file: "01_human_raw_original_all_columns.csv",
    sheet: "Human Raw",
    table: "HumanRawTable",
    widths: { A: 72, B: 14, C: 20, D: 14, E: 14, F: 10, G: 13, H: 13 },
    renderRange: "A1:H24",
  },
  {
    file: "02_human_rt_critical_with_inclusion_flags.csv",
    sheet: "Human Flags",
    table: "HumanFlagsTable",
    widths: { A: 12, B: 38, C: 14, D: 20, E: 28, F: 19, G: 14, H: 14, I: 10, J: 13, K: 13, L: 15, M: 24 },
    renderRange: "A1:M24",
  },
  {
    file: "03_rt_critical_hddm_input.csv",
    sheet: "HDDM Input",
    table: "HDDMInputTable",
    widths: { A: 14, B: 14, C: 12, D: 19, E: 14, F: 14, G: 20 },
    renderRange: "A1:G24",
  },
  {
    file: "04_rt_critical_excluded_trials.csv",
    sheet: "Excluded Trials",
    table: "ExcludedTrialsTable",
    widths: { A: 12, B: 72, C: 14, D: 20, E: 28, F: 19, G: 14, H: 14, I: 10, J: 13, K: 13, L: 24 },
    renderRange: "A1:L24",
  },
  {
    file: "05_rt_critical_subject_summary.csv",
    sheet: "Subject Summary",
    table: "SubjectSummaryTable",
    widths: { A: 14, B: 20, C: 28, D: 19, E: 12, F: 14, G: 14, H: 14, I: 17, J: 17, K: 20, L: 20, M: 14, N: 16, O: 14, P: 14 },
    renderRange: "A1:P24",
  },
  {
    file: "06_rt_critical_task_summary.csv",
    sheet: "Task Summary",
    table: "TaskSummaryTable",
    widths: { A: 20, B: 28, C: 19, D: 12, E: 12, F: 14, G: 14, H: 14, I: 17, J: 17, K: 20, L: 20, M: 14, N: 16 },
    renderRange: "A1:N6",
  },
  {
    file: "07_hddm_model_chain_stats.csv",
    sheet: "Model Chain Stats",
    table: "ModelChainStatsTable",
    widths: { A: 14, B: 10, C: 48, D: 14, E: 14, F: 14, G: 14, H: 14, I: 14, J: 14, K: 14 },
    renderRange: "A1:K24",
  },
  {
    file: "08_hddm_posterior_predictive_stats.csv",
    sheet: "PPC Stats",
    table: "PPCStatsTable",
    widths: { A: 14, B: 18, C: 14, D: 14, E: 14, F: 14, G: 14, H: 12, I: 14, J: 16 },
    renderRange: "A1:J24",
  },
  {
    file: "09_hddm_gelman_rubin.csv",
    sheet: "Convergence",
    table: "ConvergenceTable",
    widths: { A: 14, B: 52, C: 14 },
    renderRange: "A1:C24",
  },
  {
    file: "10_hddm_model_runs.csv",
    sheet: "Model Runs",
    table: "ModelRunsTable",
    widths: { A: 14, B: 10, C: 14, D: 18, E: 18, F: 16, G: 48, H: 48, I: 18 },
    renderRange: "A1:I15",
  },
  {
    file: "11_hddm_regression_posterior_summary.csv",
    sheet: "Posterior Summary",
    table: "PosteriorSummaryTable",
    widths: { A: 22, B: 20, C: 28, D: 19, E: 14, F: 14, G: 14, H: 14, I: 14 },
    renderRange: "A1:I12",
  },
  {
    file: "12_hddm_hypothesis_probabilities.csv",
    sheet: "Hypothesis Probabilities",
    table: "HypothesisProbabilitiesTable",
    widths: { A: 56, B: 22 },
    renderRange: "A1:B8",
  },
  {
    file: "13_rt_critical_effect_sizes.csv",
    sheet: "Effect Sizes",
    table: "EffectSizesTable",
    widths: { A: 14, B: 14, C: 28, D: 14, E: 14, F: 14, G: 28, H: 16, I: 14, J: 14 },
    renderRange: "A1:J10",
  },
  {
    file: "14_original_model_file_index.csv",
    sheet: "Model File Index",
    table: "ModelFileIndexTable",
    widths: { A: 58, B: 14, C: 18, D: 22, E: 52 },
    renderRange: "A1:E24",
  },
];

// Keep the largest diagnostic tables as CSV-only files to avoid Excel exporter limits.
const workbookImports = imports.filter(
  (item) => !["Human Flags", "Model Chain Stats", "Convergence"].includes(item.sheet),
);

const importedData = [];
for (const item of workbookImports) {
  const csvText = await fs.readFile(path.join(csvDir, item.file), "utf8");
  const parsedWorkbook = await Workbook.fromCSV(csvText, { sheetName: item.sheet });
  const parsedSheet = parsedWorkbook.worksheets.getItem(item.sheet);
  importedData.push({ item, values: parsedSheet.getUsedRange().values });
}

function columnName(columnCount) {
  let value = columnCount;
  let output = "";
  while (value > 0) {
    value -= 1;
    output = String.fromCharCode(65 + (value % 26)) + output;
    value = Math.floor(value / 26);
  }
  return output;
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
const summarySheet = workbook.worksheets.add("Summary");

for (const { item, values } of importedData) {
  const sheet = workbook.worksheets.add(item.sheet);
  const lastColumn = columnName(values[0].length);
  sheet.getRange(`A1:${lastColumn}${values.length}`).values = values;
}

function applyWidths(sheet, widths) {
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
}

function styleDataSheet(item) {
  const sheet = workbook.worksheets.getItem(item.sheet);
  const used = sheet.getUsedRange();
  const header = used.getRow(0);

  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Arial", size: 10, color: "#1F2933" };
  used.format.verticalAlignment = "center";
  header.format = {
    fill: "#176B67",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    rowHeight: 30,
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: "#0F4F4B" },
  };
  applyWidths(sheet, item.widths);

  const table = sheet.tables.add(used.address, true, item.table);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
}

for (const item of workbookImports) {
  styleDataSheet(item);
}

workbook.worksheets.getItem("Human Raw").getRange("D2:E36721").format.numberFormat = "0.0000";
workbook.worksheets.getItem("Human Raw").getRange("F2:F36721").format.numberFormat = "0";
workbook.worksheets.getItem("Human Raw").getRange("G2:H36721").format.numberFormat = "0.000";

const hddmInput = workbook.worksheets.getItem("HDDM Input");
hddmInput.getRange("B2:B36015").format.numberFormat = "0.0000";
hddmInput.getRange("C2:C36015").format.numberFormat = "0";
hddmInput.getRange("E2:F36015").format.numberFormat = "0.000";

const excluded = workbook.worksheets.getItem("Excluded Trials");
excluded.getRange("G2:H707").format.numberFormat = "0.0000";
excluded.getRange("I2:I707").format.numberFormat = "0";
excluded.getRange("J2:K707").format.numberFormat = "0.000";
excluded.getRange("H2:H707").conditionalFormats.add("cellIs", {
  operator: "lessThan",
  formula: 0,
  format: { fill: "#FDE2E0", font: { color: "#9B2C2C", bold: true } },
});

const subjectSummary = workbook.worksheets.getItem("Subject Summary");
subjectSummary.getRange("H2:H766").format.numberFormat = "0.00%";
subjectSummary.getRange("I2:L766").format.numberFormat = "0.0000";
subjectSummary.getRange("M2:N766").format.numberFormat = "0.00%";
subjectSummary.getRange("O2:P766").format.numberFormat = "0.000";

const taskSummary = workbook.worksheets.getItem("Task Summary");
taskSummary.getRange("H2:H4").format.numberFormat = "0.00%";
taskSummary.getRange("I2:L4").format.numberFormat = "0.0000";
taskSummary.getRange("M2:N4").format.numberFormat = "0.00%";

workbook.worksheets.getItem("PPC Stats").getRange("C2:J46").format.numberFormat = "0.000000";
workbook.worksheets.getItem("Model Runs").getRange("D2:E13").format.numberFormat = "0.000";
workbook.worksheets.getItem("Model Runs").getRange("I2:I13").format.numberFormat = "0.0000";
workbook.worksheets.getItem("Model Runs").getRange("G2:H13").format.wrapText = true;
workbook.worksheets.getItem("Model Runs").getRange("2:13").format.rowHeight = 46;
workbook.worksheets.getItem("Posterior Summary").getRange("E2:I10").format.numberFormat = "0.000000";
workbook.worksheets.getItem("Hypothesis Probabilities").getRange("B2:B10").format.numberFormat = "0.0000";
workbook.worksheets.getItem("Effect Sizes").getRange("D2:J10").format.numberFormat = "0.000";
workbook.worksheets.getItem("Model File Index").getRange("C2:C100").format.numberFormat = "#,##0";

readme.showGridLines = false;
readme.getRange("A1:H1").merge();
readme.getRange("A1").values = [["RT_critical complete data package"]];
readme.getRange("A1:H1").format = {
  fill: "#124E4A",
  font: { name: "Arial", size: 18, bold: true, color: "#FFFFFF" },
  rowHeight: 36,
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
readme.getRange("A3:B16").values = [
  ["Item", "Description"],
  ["Purpose", "Complete trial-level human RT_critical data, exact HDDM input, exclusions, summaries, and tabular model outputs."],
  ["Raw source", summary.source_file],
  ["RT source column", summary.rt_source_column],
  ["Raw human trials", summary.raw_trials],
  ["HDDM-included trials", summary.included_trials],
  ["Excluded trials", summary.excluded_trials],
  ["Exclusion rule", "RT_critical < 0; required fields must be non-missing."],
  ["Unique participant IDs", summary.subjects],
  ["Reconciliation", "The reconstructed input exactly matches the saved HDDM cleaned CSV."],
  ["Task mapping 1", "categorization / Semantic = Concept Verification"],
  ["Task mapping 2", "Voe / Intuitive = Plausibility Assessment"],
  ["Task mapping 3", "sensorimotor / Action = Affordance Recognition"],
  ["Detailed CSV-only tables", "Human inclusion flags, all chain-level parameter statistics, and all R-hat values are retained in the CSV directory to avoid Excel exporter limits."],
];
readme.getRange("A3:B3").format = {
  fill: "#176B67",
  font: { name: "Arial", size: 11, bold: true, color: "#FFFFFF" },
};
readme.getRange("A4:A16").format = {
  fill: "#E7F3F1",
  font: { name: "Arial", size: 10, bold: true, color: "#124E4A" },
};
readme.getRange("B4:B16").format = {
  font: { name: "Arial", size: 10, color: "#1F2933" },
  wrapText: true,
};
readme.getRange("A3:B16").format.borders = { preset: "insideHorizontal", style: "thin", color: "#D7E3E1" };
readme.getRange("A:A").format.columnWidth = 25;
readme.getRange("B:B").format.columnWidth = 88;
readme.getRange("3:16").format.rowHeight = 28;
readme.getRange("16:16").format.rowHeight = 48;
readme.freezePanes.freezeRows(1);

summarySheet.showGridLines = false;
summarySheet.getRange("A1:I1").merge();
summarySheet.getRange("A1").values = [["RT_critical data audit summary"]];
summarySheet.getRange("A1:I1").format = {
  fill: "#124E4A",
  font: { name: "Arial", size: 18, bold: true, color: "#FFFFFF" },
  rowHeight: 36,
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summarySheet.getRange("A3:B10").values = [
  ["Metric", "Value"],
  ["Raw human trials", null],
  ["HDDM-included trials", null],
  ["Excluded trials", null],
  ["Exclusion rate", null],
  ["Unique participant IDs", summary.subjects],
  ["Mean included RT_critical (s)", null],
  ["Included accuracy", null],
];
summarySheet.getRange("B4:B7").formulas = [
  ["=COUNTA('Human Raw'!A2:A36721)"],
  ["=COUNTA('HDDM Input'!A2:A36015)"],
  ["=COUNTA('Excluded Trials'!A2:A707)"],
  ["=B6/B4"],
];
summarySheet.getRange("B9:B10").formulas = [
  ["=AVERAGE('HDDM Input'!B2:B36015)"],
  ["=AVERAGE('HDDM Input'!C2:C36015)"],
];
summarySheet.getRange("A3:B3").format = {
  fill: "#176B67",
  font: { name: "Arial", size: 11, bold: true, color: "#FFFFFF" },
};
summarySheet.getRange("A4:A10").format = {
  fill: "#E7F3F1",
  font: { name: "Arial", size: 10, bold: true, color: "#124E4A" },
};
summarySheet.getRange("B4:B10").format.font = { name: "Arial", size: 11, color: "#1F2933" };
summarySheet.getRange("B4:B6").format.numberFormat = "#,##0";
summarySheet.getRange("B7:B7").format.numberFormat = "0.00%";
summarySheet.getRange("B8:B8").format.numberFormat = "#,##0";
summarySheet.getRange("B9:B9").format.numberFormat = "0.0000";
summarySheet.getRange("B10:B10").format.numberFormat = "0.00%";

summarySheet.getRange("A13:I17").values = [
  ["Condition", "Task name", "Raw trials", "Included trials", "Excluded trials", "Exclusion rate", "Mean raw RT", "Mean included RT", "Included accuracy"],
  ["categorization", "Concept Verification", null, null, null, null, null, null, null],
  ["Voe", "Plausibility Assessment", null, null, null, null, null, null, null],
  ["sensorimotor", "Affordance Recognition", null, null, null, null, null, null, null],
  ["Total", "All tasks", null, null, null, null, null, null, null],
];

for (let row = 14; row <= 16; row += 1) {
  summarySheet.getRange(`C${row}:I${row}`).formulas = [[
    `=COUNTIF('Human Raw'!$C$2:$C$36721,A${row})`,
    `=COUNTIF('HDDM Input'!$G$2:$G$36015,A${row})`,
    `=C${row}-D${row}`,
    `=E${row}/C${row}`,
    `=AVERAGEIF('Human Raw'!$C$2:$C$36721,A${row},'Human Raw'!$E$2:$E$36721)`,
    `=AVERAGEIF('HDDM Input'!$G$2:$G$36015,A${row},'HDDM Input'!$B$2:$B$36015)`,
    `=AVERAGEIF('HDDM Input'!$G$2:$G$36015,A${row},'HDDM Input'!$C$2:$C$36015)`,
  ]];
}
summarySheet.getRange("C17:I17").formulas = [[
  "=SUM(C14:C16)",
  "=SUM(D14:D16)",
  "=SUM(E14:E16)",
  "=E17/C17",
  "=AVERAGE('Human Raw'!E2:E36721)",
  "=AVERAGE('HDDM Input'!B2:B36015)",
  "=AVERAGE('HDDM Input'!C2:C36015)",
]];
summarySheet.getRange("A13:I13").format = {
  fill: "#176B67",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  wrapText: true,
  rowHeight: 32,
};
summarySheet.getRange("A14:B17").format.font = { name: "Arial", size: 10, color: "#1F2933" };
summarySheet.getRange("A17:I17").format = {
  fill: "#E7F3F1",
  font: { name: "Arial", size: 10, bold: true, color: "#124E4A" },
};
summarySheet.getRange("C14:E17").format.numberFormat = "#,##0";
summarySheet.getRange("F14:F17").format.numberFormat = "0.00%";
summarySheet.getRange("G14:H17").format.numberFormat = "0.0000";
summarySheet.getRange("I14:I17").format.numberFormat = "0.00%";
summarySheet.getRange("A:A").format.columnWidth = 32;
summarySheet.getRange("B:B").format.columnWidth = 29;
summarySheet.getRange("C:E").format.columnWidth = 15;
summarySheet.getRange("F:F").format.columnWidth = 16;
summarySheet.getRange("G:I").format.columnWidth = 18;
summarySheet.freezePanes.freezeRows(1);

if (process.env.SKIP_WORKBOOK_RENDER !== "1") {
  await fs.mkdir(qaDir, { recursive: true });
  const allRenderJobs = [
    { sheet: "README", range: "A1:B16" },
    { sheet: "Summary", range: "A1:I17" },
    ...workbookImports.map((item) => ({ sheet: item.sheet, range: item.renderRange })),
  ];
  const renderJobs = process.env.RENDER_CHANGED_SHEETS_ONLY === "1"
    ? allRenderJobs.filter((job) => ["README", "Summary", "Human Raw", "Excluded Trials", "Model Runs"].includes(job.sheet))
    : allRenderJobs;

  for (const job of renderJobs) {
    const preview = await workbook.render({
      sheetName: job.sheet,
      range: job.range,
      scale: 1.2,
      format: "png",
    });
    const filename = `${job.sheet.replaceAll(" ", "_")}.png`;
    await fs.writeFile(path.join(qaDir, filename), new Uint8Array(await preview.arrayBuffer()));
  }
}

const summaryInspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:I17",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 12,
  maxChars: 8000,
});
console.log(summaryInspection.ndjson);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);

if (process.env.SKIP_WORKBOOK_EXPORT !== "1") {
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(path.join(outputDir, "RT_critical_complete_data.xlsx"));
  console.log(path.join(outputDir, "RT_critical_complete_data.xlsx"));
}
console.log(qaDir);
