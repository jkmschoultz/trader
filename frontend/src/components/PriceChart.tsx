import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  type IChartApi,
  type CandlestickData,
  type UTCTimestamp,
  type SeriesMarker,
} from "lightweight-charts";

import type { Bar, Fill } from "../api/types";

interface Props {
  bars: Bar[];
  fills?: Fill[];
  height?: number;
}

const dark = () =>
  window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

export function PriceChart({ bars, fills = [], height = 420 }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!el.current) return;
    const isDark = dark();
    const chart = createChart(el.current, {
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
      rightPriceScale: { borderColor: isDark ? "#334155" : "#cbd5e1" },
    });
    chartRef.current = chart;

    const candles = chart.addCandlestickSeries({
      upColor: "#16a34a",
      downColor: "#dc2626",
      wickUpColor: "#16a34a",
      wickDownColor: "#dc2626",
      borderVisible: false,
    });
    const volume = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      color: isDark ? "#334155" : "#cbd5e1",
    });
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    const candleData: CandlestickData[] = bars
      .filter((b) => b.open != null && b.high != null && b.low != null && b.close != null)
      .map((b) => ({
        time: b.time as UTCTimestamp,
        open: b.open as number,
        high: b.high as number,
        low: b.low as number,
        close: b.close as number,
      }));
    candles.setData(candleData);
    volume.setData(
      bars
        .filter((b) => b.volume != null)
        .map((b) => ({ time: b.time as UTCTimestamp, value: b.volume as number })),
    );

    if (fills.length) {
      const markers: SeriesMarker<UTCTimestamp>[] = fills
        .map((f) => {
          const buy = f.units > 0;
          return {
            time: Math.floor(new Date(f.time).getTime() / 1000) as UTCTimestamp,
            position: buy ? ("belowBar" as const) : ("aboveBar" as const),
            color: buy ? "#16a34a" : "#dc2626",
            shape: buy ? ("arrowUp" as const) : ("arrowDown" as const),
            text: `${buy ? "+" : ""}${f.units}`,
          };
        })
        .sort((a, b) => (a.time as number) - (b.time as number));
      candles.setMarkers(markers);
    }

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (el.current) chart.applyOptions({ width: el.current.clientWidth });
    });
    ro.observe(el.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, fills, height]);

  return <div ref={el} className="w-full" />;
}
