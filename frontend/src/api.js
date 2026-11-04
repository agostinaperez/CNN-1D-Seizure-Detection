export async function fetchModels() {
  const r = await fetch('/api/models')
  if (!r.ok) throw new Error('No se pudo cargar la lista de modelos')
  return r.json()
}

export async function fetchModelInfo(name) {
  const r = await fetch('/api/models/' + encodeURIComponent(name))
  if (!r.ok) throw new Error('No se pudo cargar la información del modelo')
  return r.json()
}

export async function runInference(model, file) {
  const fd = new FormData()
  fd.append('model', model)
  fd.append('file', file)
  const r = await fetch('/api/infer', { method: 'POST', body: fd })
  if (!r.ok) {
    let detail = 'Error en la inferencia'
    try {
      detail = (await r.json()).detail || detail
    } catch (_) {}
    throw new Error(detail)
  }
  return r.json()
}
