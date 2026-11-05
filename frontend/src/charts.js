import * as echarts from 'echarts'

const AXIS = '#99a29b'
const GRID_LINE = 'rgba(153, 162, 155, 0.08)'
const GREEN = '#8aa86f'
const GREEN_DARK = '#6d8f56'
const ROSE = '#e05f5f'
const AMBER = '#d9a24a'

function sliderZoom() {
  return {
    type: 'slider',
    xAxisIndex: 0,
    height: 18,
    bottom: 2,
    borderColor: '#252d27',
    backgroundColor: '#161b18',
    fillerColor: 'rgba(109, 143, 86, 0.15)',
    handleStyle: { color: '#6d8f56' },
    moveHandleStyle: { color: '#6d8f56' },
    textStyle: { color: '#99a29b' },
    dataBackground: { lineStyle: { color: '#3a463c' }, areaStyle: { color: '#222a24' } },
  }
}

export function probabilityOption(res, threshold) {
  const data = res.window_starts.map((s, i) => [s, res.probabilities[i]])
  const markAreas = res.events.map((e) => [
    { name: 'Crisis', xAxis: e.start_s, itemStyle: { color: 'rgba(224, 95, 95, 0.14)' } },
    { xAxis: e.end_s },
  ])

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1c231e',
      borderColor: '#252d27',
      textStyle: { color: '#e7eae7' },
      valueFormatter: (v) => (v == null ? '—' : Number(v).toFixed(3)),
    },
    grid: { left: 50, right: 24, top: 28, bottom: 52 },
    xAxis: {
      type: 'value',
      name: 'Tiempo (s)',
      nameTextStyle: { color: AXIS },
      axisLine: { lineStyle: { color: GRID_LINE } },
      axisLabel: { color: AXIS },
      splitLine: { lineStyle: { color: GRID_LINE } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: 'P(crisis)',
      nameTextStyle: { color: AXIS },
      axisLabel: { color: AXIS },
      splitLine: { lineStyle: { color: GRID_LINE } },
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, throttle: 50 },
      sliderZoom(),
    ],
    series: [
      {
        name: 'P(crisis)',
        type: 'line',
        showSymbol: false,
        smooth: false,
        data,
        lineStyle: { color: GREEN, width: 1.6 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(138, 168, 111, 0.22)' },
            { offset: 1, color: 'rgba(138, 168, 111, 0.02)' },
          ]),
        },
        markLine: {
          symbol: 'none',
          silent: true,
          data: [
            {
              yAxis: threshold,
              label: { formatter: `umbral ${threshold}`, color: AMBER, position: 'insideEndTop' },
              lineStyle: { color: AMBER, type: 'dashed', width: 1 },
            },
          ],
        },
        markArea: { data: markAreas },
      },
    ],
  }
}

export function signalOption(signal, events) {
  const channels = signal.channels
  const sfreq = signal.sfreq
  const n = channels.length
  const tMax = signal.n_samples / sfreq

  const grids = []
  const xAxes = []
  const yAxes = []
  const series = []
  const xIndexes = []

  const usable = 92 // porcentaje de alto reservado a los canales (el resto, al slider)
  const step = usable / n
  const markAreas = events.map((e) => [
    { name: 'Crisis', xAxis: e.start_s, itemStyle: { color: 'rgba(224, 95, 95, 0.12)' } },
    { xAxis: e.end_s },
  ])

  for (let c = 0; c < n; c++) {
    const top = (c * step + 0.2).toFixed(2)
    const height = Math.max(step - 0.4, 0.1).toFixed(2)
    grids.push({ left: 70, right: 24, top: `${top}%`, height: `${height}%` })
    xIndexes.push(c)

    xAxes.push({
      type: 'value',
      gridIndex: c,
      min: 0,
      max: tMax,
      axisLabel: { show: false },
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
    })

    yAxes.push({
      type: 'value',
      gridIndex: c,
      name: channels[c],
      nameLocation: 'middle',
      nameGap: 58,
      nameTextStyle: { color: '#c7cec8', fontSize: 10, align: 'right' },
      axisLabel: { show: false },
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
    })

    series.push({
      type: 'line',
      xAxisIndex: c,
      yAxisIndex: c,
      showSymbol: false,
      sampling: 'lttb',
      data: signal.data[c].map((v, i) => [i / sfreq, v]),
      lineStyle: { color: c === 0 ? GREEN_DARK : '#3f5a38', width: 0.6, opacity: 0.95 },
      markArea: { silent: true, data: markAreas },
    })
  }

  return {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1c231e',
      borderColor: '#252d27',
      textStyle: { color: '#e7eae7' },
    },
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    dataZoom: [
      { type: 'inside', xAxisIndex: xIndexes, throttle: 50 },
      { type: 'slider', xAxisIndex: xIndexes, height: 18, bottom: 2, borderColor: '#252d27', backgroundColor: '#161b18', fillerColor: 'rgba(109, 143, 86, 0.15)', handleStyle: { color: '#6d8f56' }, moveHandleStyle: { color: '#6d8f56' }, textStyle: { color: '#99a29b' }, dataBackground: { lineStyle: { color: '#3a463c' }, areaStyle: { color: '#222a24' } } },
    ],
    series,
  }
}

export function historyOption(history) {
  const epochs = history.map((h) => h.epoch)
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1c231e',
      borderColor: '#252d27',
      textStyle: { color: '#e7eae7' },
    },
    legend: {
      textStyle: { color: AXIS, fontSize: 11 },
      top: 0,
    },
    grid: { left: 46, right: 46, top: 36, bottom: 30 },
    xAxis: {
      type: 'category',
      data: epochs,
      name: 'Época',
      nameTextStyle: { color: AXIS },
      axisLabel: { color: AXIS },
      axisLine: { lineStyle: { color: GRID_LINE } },
    },
    yAxis: [
      {
        type: 'value',
        name: 'Loss',
        nameTextStyle: { color: AXIS },
        axisLabel: { color: AXIS },
        splitLine: { lineStyle: { color: GRID_LINE } },
      },
      {
        type: 'value',
        name: 'AUPRC / AUC',
        min: 0,
        max: 1,
        nameTextStyle: { color: AXIS },
        axisLabel: { color: AXIS },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: 'train loss',
        type: 'line',
        showSymbol: false,
        itemStyle: { color: GREEN_DARK },
        lineStyle: { color: GREEN_DARK, width: 1.4 },
        data: history.map((h) => h.train_loss),
      },
      {
        name: 'val loss',
        type: 'line',
        showSymbol: false,
        itemStyle: { color: '#4d8db8' },
        lineStyle: { color: '#4d8db8', width: 1.4 },
        data: history.map((h) => h.val_loss),
      },
      {
        name: 'val AUPRC',
        type: 'line',
        yAxisIndex: 1,
        showSymbol: false,
        itemStyle: { color: AMBER },
        lineStyle: { color: AMBER, width: 1.2, type: 'dashed' },
        data: history.map((h) => h.val_auprc),
      },
      {
        name: 'val AUC-ROC',
        type: 'line',
        yAxisIndex: 1,
        showSymbol: false,
        itemStyle: { color: '#7aa6a0' },
        lineStyle: { color: '#7aa6a0', width: 1.2, type: 'dashed' },
        data: history.map((h) => h.val_auc_roc),
      },
    ],
  }
}
