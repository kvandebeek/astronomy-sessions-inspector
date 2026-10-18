// Lint config for the pages' browser scripts in static/ (npx eslint static).
// The scripts are concatenated into one inline <script> by the Python side, so each sees
// the others' top-level functions; those - and the inline onclick handlers' targets - are
// declared here rather than flagged as undefined or unused.
export default [
  {
    files: ["static/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        window: "readonly", document: "readonly", console: "readonly", fetch: "readonly",
        alert: "readonly", confirm: "readonly", prompt: "readonly", location: "readonly",
        setTimeout: "readonly", setInterval: "readonly", requestAnimationFrame: "readonly",
        getComputedStyle: "readonly", Image: "readonly", Blob: "readonly", URL: "readonly",
        ResizeObserver: "readonly", IntersectionObserver: "readonly",
        // Provided by the page (embedded data) or by a sibling script.
        REVIEW_DATA: "readonly", COMP: "readonly", COMP_METRICS: "readonly",
        rowTotalKey: "readonly", setRowTotal: "readonly",
      },
    },
    rules: {
      "no-undef": "error",
      "no-unused-vars": ["error", {vars: "local", args: "none", caughtErrors: "none"}],
      "no-redeclare": ["error", {builtinGlobals: false}],
      "no-dupe-keys": "error",
      "no-unreachable": "error",
      "eqeqeq": ["error", "smart"],
    },
  },
];
