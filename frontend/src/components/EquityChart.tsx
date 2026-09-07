import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  LineStyle,
  type IChartApi,
  type UTCTimestamp,
} from "lightweight-charts";

import type { EquityPoint } from "../api/types";

const dark = () =>
  window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

export function EquityChart({ equity, height = 300 }: { equity: EquityPoint[]; height?: number }) {
  const el = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!el.current || equity.length === 0) return;
    const isDark = dark();
    const chart: IChartApi = createChart(el.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: isDark ? "#cbd5e1" : "#334155",
      },
      grid: {
        vertLines: { color: isDark ? "#1e293b" : "#e2e8f0" },
        horzLines: { color: isDark ? "#1e293b" : "#e2e8f0" },
      },
      timeScale: { timeVisible: true, secondsVisible: false },
    });

    const points = equity.map((p) => ({
      time: Math.floor(new Date(p.time).getTime() / 1000) as UTCTimestamp,
      value: p.equity,
    }));

    const area = chart.addAreaSeries({
      lineColor: "#2563eb",
      topColor: "rgba(37, 99, 235, 0.4)",
      bottomColor: "rgba(37, 99, 235, 0.02)",
      lineWidth: 2,
    });
    area.setData(points);

    // drawdown as a percentage line on a hidden left scale
    let peak = -Infinity;
    const dd = points.map((p) => {
      peak = Math.max(peak, p.value);
      return { time: p.time, value: peak > 0 ? (p.value / peak - 1) * 100 : 0 };
    });
    const ddSeries = chart.addLineSeries({
      color: isDark ? "#f87171" : "#ef4444",
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceScaleId: "left",
      priceFormat: { type: "custom", formatter: (v: number) => `${v.toFixed(1)}%` },
    });
    ddSeries.setData(dd);
    chart.priceScale("left").applyOptions({ visible: true });

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (el.current) chart.applyOptions({ width: el.current.clientWidth });
    });
    ro.observe(el.current);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [equity, height]);

  return <div ref={el} className="w-full" />;
}
