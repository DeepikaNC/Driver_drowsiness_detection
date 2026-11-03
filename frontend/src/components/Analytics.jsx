
import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Activity, Eye, Gauge, AlertTriangle } from "lucide-react";
import "./Analytics.css";

const API = "http://127.0.0.1:8000";

function Analytics() {
  const [history, setHistory] = useState([]);

  const [status, setStatus] = useState({
    ear: null,
    mar: null,
    drowsy_probability: null,
    recent_yawns: 0,
    model_status: "Warming up",
    warning: false,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(
          `${API}/api/camera/status`
        );

        if (!response.ok) return;

        const data = await response.json();

        setStatus(data);

        if (
          Number.isFinite(data.ear) &&
          Number.isFinite(data.mar)
        ) {
          setHistory((previous) => [
            ...previous,
            {
              time: new Date().toLocaleTimeString(),
              ear: data.ear,
              mar: data.mar,
              probability:
                Number.isFinite(data.drowsy_probability)
                  ? data.drowsy_probability * 100
                  : null,
            },
          ].slice(-30));
        }
      } catch (error) {
        console.error("Analytics error:", error);
      }
    };

    fetchData();

    const interval = setInterval(fetchData, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="analytics-page">

      <div className="analytics-heading">
        <div>
          <h2>Driver Performance Analytics</h2>
          <p>Live facial measurements and model predictions</p>
        </div>

        <span className="analytics-live">
          <span /> LIVE DATA
        </span>
      </div>

      <div className="analytics-stats">

        <div className="analytics-card">
          <Eye />
          <p>Current EAR</p>
          <h2>
            {Number.isFinite(status.ear)
              ? status.ear.toFixed(3)
              : "--"}
          </h2>
        </div>

        <div className="analytics-card">
          <Activity />
          <p>Current MAR</p>
          <h2>
            {Number.isFinite(status.mar)
              ? status.mar.toFixed(3)
              : "--"}
          </h2>
        </div>

        <div className="analytics-card">
          <Gauge />
          <p>Drowsiness Probability</p>
          <h2>
            {Number.isFinite(status.drowsy_probability)
              ? `${(status.drowsy_probability * 100).toFixed(1)}%`
              : "--"}
          </h2>
        </div>

        <div className="analytics-card">
          <AlertTriangle />
          <p>Recent Yawns</p>
          <h2>{status.recent_yawns ?? 0}</h2>
        </div>

      </div>

      <div className="analytics-chart-grid">

        <div className="analytics-panel">
          <h3>Eye Aspect Ratio (EAR)</h3>
          <p>Eye openness measurement over time</p>

          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={history}>
              <CartesianGrid stroke="#29374e" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="#8291a8" />
              <YAxis stroke="#8291a8" />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="ear"
                name="EAR"
                stroke="#48d5e5"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="analytics-panel">
          <h3>Mouth Aspect Ratio (MAR)</h3>
          <p>Mouth movement measurement over time</p>

          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={history}>
              <CartesianGrid stroke="#29374e" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="#8291a8" />
              <YAxis stroke="#8291a8" />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="mar"
                name="MAR"
                stroke="#b69aff"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="analytics-panel">
          <h3>Drowsiness Probability</h3>
          <p>Random Forest model probability of Drowsy class</p>

          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={history}>
              <CartesianGrid stroke="#29374e" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="#8291a8" />
              <YAxis domain={[0, 100]} stroke="#8291a8" />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="probability"
                name="Drowsiness %"
                stroke="#f2a76e"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="analytics-panel">
          <h3>Current Driver Assessment</h3>
          <p>Latest classification from the model</p>

          <div className="assessment-box">
            <h2 className={status.warning ? "warning" : ""}>
              {status.warning
                ? "DROWSINESS WARNING"
                : status.model_status}
            </h2>

            <p>
              Recent yawns: {status.recent_yawns ?? 0} / 3
            </p>

            <p>
              Face detected: {status.face_detected ? "Yes" : "No"}
            </p>
          </div>
        </div>

      </div>

    </div>
  );
}

export default Analytics;
