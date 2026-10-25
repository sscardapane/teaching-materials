// One Pyodide runtime (with NumPy) per page, shared by PyRunner and QuizCode.
// Loaded from the CDN on first use (~10 MB of WASM).
const VERSION = 'v0.26.4'
let runtime = null

export function loadPython() {
  if (!runtime) {
    runtime = (async () => {
      if (!window.loadPyodide) {
        await new Promise((resolve, reject) => {
          const script = document.createElement('script')
          script.src = `https://cdn.jsdelivr.net/pyodide/${VERSION}/full/pyodide.js`
          script.onload = resolve
          script.onerror = reject
          document.head.appendChild(script)
        })
      }
      const py = await window.loadPyodide({
        indexURL: `https://cdn.jsdelivr.net/pyodide/${VERSION}/full/`,
      })
      await py.loadPackage('numpy')
      return py
    })()
    runtime.catch(() => { runtime = null })
  }
  return runtime
}
