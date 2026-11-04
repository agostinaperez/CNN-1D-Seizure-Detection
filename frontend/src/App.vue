<script setup>
import { ref, computed, onMounted } from 'vue'
import EChart from './components/EChart.vue'
import { fetchModels, fetchModelInfo, runInference } from './api'
import { probabilityOption, signalOption, historyOption } from './charts'

const models = ref([])
const selectedModel = ref('')
const modelInfo = ref(null)
const loadingModel = ref(false)

const file = ref(null)
const fileLabel = ref('')
const running = ref(false)
const result = ref(null)
const error = ref('')
const modelsError = ref('')
const dragging = ref(false)
const fullscreen = ref(null)

const probOption = computed(() => (result.value ? probabilityOption(result.value, result.value.threshold) : null))
const signalOpt = computed(() => (result.value ? signalOption(result.value.signal, result.value.events) : null))
const histOption = computed(() => (modelInfo.value && modelInfo.value.history ? historyOption(modelInfo.value.history) : null))
const signalHeight = computed(() => {
  const n = result.value ? result.value.signal.channels.length : 16
  return `${Math.max(n * 26 + 90, 460)}px`
})
const fullscreenHeight = 'calc(100vh - 130px)'

async function loadModels() {
  try {
    const data = await fetchModels()
    models.value = data.models || []
    if (models.value.length) {
      selectedModel.value = models.value[0].name
      await selectModel()
    }
  } catch (e) {
    modelsError.value = e.message
  }
}

async function selectModel() {
  if (!selectedModel.value) return
  loadingModel.value = true
  error.value = ''
  try {
    modelInfo.value = await fetchModelInfo(selectedModel.value)
  } catch (e) {
    error.value = e.message
  } finally {
    loadingModel.value = false
  }
}

function onFileChange(ev) {
  const f = ev.target.files && ev.target.files[0]
  if (f) setFile(f)
}

function setFile(f) {
  file.value = f
  fileLabel.value = f.name
  result.value = null
  error.value = ''
}

function onDrop(ev) {
  dragging.value = false
  const f = ev.dataTransfer.files && ev.dataTransfer.files[0]
  if (f) setFile(f)
}

async function doInference() {
  if (!file.value || !selectedModel.value) return
  running.value = true
  error.value = ''
  try {
    result.value = await runInference(selectedModel.value, file.value)
  } catch (e) {
    error.value = e.message
  } finally {
    running.value = false
  }
}

function openFullscreen(title, option) {
  fullscreen.value = { title, option }
}

function closeFullscreen() {
  fullscreen.value = null
}

function fmtDur(s) {
  s = Number(s || 0)
  if (s >= 3600) {
    const h = Math.floor(s / 3600)
    const m = Math.round((s % 3600) / 60)
    return `${h} h ${m} min`
  }
  if (s >= 60) {
    const m = Math.floor(s / 60)
    const r = Math.round(s % 60)
    return `${m} min ${r} s`
  }
  return `${s.toFixed(1).replace('.', ',')} s`
}

onMounted(loadModels)
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <h1>Detección de crisis epilépticas</h1>
        <p>CNN-1D sobre EEG · montaje TUEV · CHB-MIT</p>
      </div>
      <div class="topbar-info" v-if="modelInfo">
        <span class="mono">{{ selectedModel }}</span>
        <span class="sep">·</span>
        <span>umbral {{ modelInfo.threshold }}</span>
      </div>
    </header>

    <main class="layout">
      <aside class="sidebar">
        <section class="card">
          <h2 class="card-title">Modelo</h2>
          <p class="muted small" v-if="modelsError">{{ modelsError }}</p>
          <select v-model="selectedModel" class="select" @change="selectModel">
            <option v-for="m in models" :key="m.name" :value="m.name">
              {{ m.name }} — {{ m.loss }} · ep {{ m.epoch }}
            </option>
          </select>

          <div class="kv" v-if="modelInfo">
            <div class="kv-row"><span>Loss</span><b>{{ modelInfo.loss }}</b></div>
            <div class="kv-row"><span>Focal α / γ</span><b>{{ modelInfo.focal_alpha }} / {{ modelInfo.focal_gamma }}</b></div>
            <div class="kv-row"><span>Época guardada</span><b>{{ modelInfo.epoch }}</b></div>
            <div class="kv-row"><span>Umbral operativo</span><b>{{ modelInfo.threshold }}</b></div>
            <div class="kv-row"><span>Criterio</span><b>{{ modelInfo.selection_metric }}</b></div>
          </div>
          <div class="loading" v-if="loadingModel">Cargando modelo…</div>
        </section>

        <section class="card" v-if="histOption">
          <div class="card-head">
            <h2 class="card-title">Entrenamiento</h2>
            <button class="icon-btn" title="Pantalla completa" @click="openFullscreen('Historial de entrenamiento', histOption)">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
            </button>
          </div>
          <EChart :option="histOption" height="260px" />
        </section>
      </aside>

      <section class="content">
        <section class="card">
          <h2 class="card-title">Inferencia</h2>
          <div
            class="dropzone"
            :class="{ active: dragging, has: !!file }"
            @dragover.prevent="dragging = true"
            @dragleave.prevent="dragging = false"
            @drop.prevent="onDrop"
          >
            <input id="edf-input" type="file" accept=".edf,.EDF" class="hidden-input" @change="onFileChange" />
            <label for="edf-input" class="drop-label">
              <div class="drop-text">
                <template v-if="file">
                  <b>{{ file.name }}</b>
                  <span class="muted">{{ (file.size / 1048576).toFixed(1).replace('.', ',') }} MB · clic para cambiar</span>
                </template>
                <template v-else>
                  <b>Seleccionar archivo .edf</b>
                  <span class="muted">arrastralo acá o hacé clic para elegirlo</span>
                </template>
              </div>
            </label>
          </div>

          <button class="btn" :disabled="!file || running || loadingModel" @click="doInference">
            <span v-if="running" class="spinner"></span>
            {{ running ? 'Inferiendo…' : 'Ejecutar inferencia' }}
          </button>

          <div class="error" v-if="error">{{ error }}</div>
        </section>

        <template v-if="result">
          <section class="summary">
            <div class="summary-cell"><span class="summary-v">{{ fmtDur(result.duration_s) }}</span><span class="summary-l">duración</span></div>
            <div class="summary-cell"><span class="summary-v">{{ result.n_windows }}</span><span class="summary-l">ventanas</span></div>
            <div class="summary-cell" :class="{ alert: result.n_events > 0 }">
              <span class="summary-v">{{ result.n_events }}</span>
              <span class="summary-l">eventos detectados</span>
            </div>
            <div class="summary-cell"><span class="summary-v">{{ result.threshold }}</span><span class="summary-l">umbral</span></div>
            <div class="summary-cell"><span class="summary-v">{{ result.inference_time_s.toFixed(2).replace('.', ',') }} s</span><span class="summary-l">tiempo de inferencia</span></div>
          </section>

          <section class="card">
            <div class="card-head">
              <h2 class="card-title">Probabilidad de crisis por ventana</h2>
              <button class="icon-btn" title="Pantalla completa" @click="openFullscreen('Probabilidad de crisis', probOption)">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
              </button>
            </div>
            <EChart v-if="probOption" :option="probOption" height="320px" />
          </section>

          <section class="card">
            <div class="card-head">
              <h2 class="card-title">Señal EEG ({{ result.signal.channels.length }} canales)</h2>
              <button class="icon-btn" title="Pantalla completa" @click="openFullscreen('Señal EEG', signalOpt)">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
              </button>
            </div>
            <EChart v-if="signalOpt" :option="signalOpt" :height="signalHeight" />
          </section>

          <section class="card">
            <h2 class="card-title">Eventos detectados</h2>
            <p v-if="!result.events.length" class="muted">No se detectaron crisis epilépticas en este registro.</p>
            <table v-else class="events">
              <thead>
                <tr><th>#</th><th>Inicio</th><th>Fin</th><th>Duración</th></tr>
              </thead>
              <tbody>
                <tr v-for="(e, i) in result.events" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td class="mono">{{ e.start_s.toFixed(1).replace('.', ',') }} s</td>
                  <td class="mono">{{ e.end_s.toFixed(1).replace('.', ',') }} s</td>
                  <td class="mono">{{ e.duration_s.toFixed(1).replace('.', ',') }} s</td>
                </tr>
              </tbody>
            </table>
          </section>
        </template>

        <div class="placeholder" v-else-if="!result">
          <p>Elegí un modelo, subí un EDF y ejecutá la inferencia para ver los resultados.</p>
        </div>
      </section>
    </main>

    <div class="overlay" v-if="fullscreen" @click.self="closeFullscreen">
      <div class="fs-modal">
        <div class="fs-head">
          <span>{{ fullscreen.title }}</span>
          <button class="icon-btn" @click="closeFullscreen">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
          </button>
        </div>
        <div class="fs-body">
          <EChart :option="fullscreen.option" :height="fullscreenHeight" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 28px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  position: sticky;
  top: 0;
  z-index: 10;
}

.brand h1 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.2px;
}

.brand p {
  margin: 2px 0 0;
  color: var(--muted);
  font-size: 12px;
}

.topbar-info {
  font-size: 13px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 8px;
}

.topbar-info .sep {
  color: var(--border);
}

.mono {
  font-family: 'SF Mono', 'Cascadia Mono', Consolas, monospace;
  font-variant-numeric: tabular-nums;
}

.layout {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 16px;
  padding: 16px 28px 60px;
  max-width: 1500px;
  margin: 0 auto;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}

.card-title {
  margin: 0;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--muted);
}

.card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.icon-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--muted);
  border-radius: 6px;
  width: 28px;
  height: 28px;
  cursor: pointer;
  font-size: 13px;
  line-height: 1;
  display: grid;
  place-items: center;
}

.icon-btn:hover {
  color: var(--text);
  border-color: var(--accent);
}

.muted {
  color: var(--muted);
}

.small {
  font-size: 12px;
}

.select {
  width: 100%;
  background: var(--panel-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 9px 10px;
  font-size: 13px;
  margin-bottom: 12px;
  cursor: pointer;
}

.kv {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.kv-row {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  gap: 12px;
}

.kv-row span {
  color: var(--muted);
}

.kv-row b {
  text-align: right;
  font-weight: 500;
}

.loading {
  color: var(--muted);
  font-size: 13px;
  margin-top: 10px;
}

.dropzone {
  border: 1px dashed var(--border);
  border-radius: 6px;
  transition: border-color 0.15s, background 0.15s;
  cursor: pointer;
}

.dropzone.active {
  border-color: var(--accent);
  background: rgba(109, 143, 86, 0.06);
}

.dropzone.has {
  border-style: solid;
  border-color: var(--accent);
}

.hidden-input {
  display: none;
}

.drop-label {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 18px;
  cursor: pointer;
}

.drop-text {
  display: flex;
  flex-direction: column;
  gap: 3px;
  font-size: 13px;
}

.btn {
  margin-top: 12px;
  width: 100%;
  background: var(--accent);
  color: #0e1211;
  border: none;
  border-radius: 6px;
  padding: 12px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  transition: filter 0.1s, opacity 0.2s;
}

.btn:hover:not(:disabled) {
  filter: brightness(1.08);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.spinner {
  width: 15px;
  height: 15px;
  border: 2px solid rgba(14, 18, 17, 0.3);
  border-top-color: #0e1211;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error {
  margin-top: 12px;
  color: #f0a1a1;
  font-size: 13px;
  background: rgba(224, 95, 95, 0.08);
  border: 1px solid rgba(224, 95, 95, 0.35);
  padding: 10px 12px;
  border-radius: 6px;
}

.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}

.summary-cell {
  background: var(--panel);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.summary-v {
  font-size: 20px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.summary-l {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.summary-cell.alert .summary-v {
  color: var(--rose);
}

.events {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.events th,
.events td {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}

.events th {
  color: var(--muted);
  font-weight: 500;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.events tr:last-child td {
  border-bottom: none;
}

.placeholder {
  color: var(--muted);
  text-align: center;
  padding: 48px 20px;
  border: 1px dashed var(--border);
  border-radius: var(--radius);
}

.overlay {
  position: fixed;
  inset: 0;
  background: rgba(8, 10, 9, 0.88);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.fs-modal {
  width: 100%;
  max-width: 1400px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.fs-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 12px;
  font-size: 14px;
  font-weight: 600;
}

.fs-body {
  flex: 1;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px;
  min-height: 0;
}
</style>
