import Analytics from "./components/Analytics";
import { useState, useEffect } from "react";

import {
  Activity,
  AlertTriangle,
  Bell,
  Camera,
  CarFront,
  CheckCircle2,
  Clock3,
  Cpu,
  Eye,
  Gauge,
  LayoutDashboard,
  Mic,
  Settings,
  ShieldCheck,
  Video,
  Wifi,
  WifiOff,
  Zap,
  Square,
} from "lucide-react";

import "./App.css";

const API = "http://127.0.0.1:8000";

const initialCameraStatus = {
  camera: "Not started",
  face_detected: false,
  model_status: "Warming up",
  ear: null,
  mar: null,
  drowsy_probability: null,
  recent_yawns: 0,
  warning: false,
  message: "Camera is not started",
};

function App() {
  const [activePage, setActivePage] = useState("Dashboard");

  const [backendOnline, setBackendOnline] = useState(false);

  const [cameraStatus, setCameraStatus] = useState(
    initialCameraStatus
  );
  const [voiceStatus, setVoiceStatus] = useState({
    status: "Idle",
    message: "Voice assistant is waiting",
    active: false,
  });
  const [arduinoStatus, setArduinoStatus] = useState({
    connected: false,
    status: "Disconnected",
    alert_active: false,
    port: "COM7",
  });

  const [cameraError, setCameraError] = useState("");

  const [loading, setLoading] = useState(false);

  const [streamVersion, setStreamVersion] = useState(0);

  // -----------------------------------------
  // ARDUINO CONNECTION AND LIVE STATUS
  // -----------------------------------------

  useEffect(() => {
    let mounted = true;

    const updateArduinoStatus = async () => {
      try {
        const response = await fetch(`${API}/api/arduino/status`);
        if (!response.ok) {
          throw new Error("Unable to retrieve Arduino status");
        }

        const data = await response.json();
        if (mounted) {
          setArduinoStatus((previous) => ({ ...previous, ...data }));
        }
      } catch (error) {
        if (mounted) {
          setArduinoStatus((previous) => ({
            ...previous,
            connected: false,
            status: "Disconnected",
            alert_active: false,
          }));
        }
        console.error("Arduino status error:", error);
      }
    };

    const connectArduino = async () => {
      try {
        const response = await fetch(`${API}/api/arduino/connect`, {
          method: "POST",
        });
        const data = await response.json();

        if (mounted) {
          setArduinoStatus((previous) => ({ ...previous, ...data }));
        }
      } catch (error) {
        console.error("Arduino connection error:", error);
        if (mounted) {
          setArduinoStatus((previous) => ({
            ...previous,
            connected: false,
            status: "Disconnected",
            alert_active: false,
          }));
        }
      }
    };

    connectArduino();
    updateArduinoStatus();

    const interval = setInterval(updateArduinoStatus, 1500);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // -----------------------------------------
  // BACKEND HEALTH CHECK
  // -----------------------------------------


  useEffect(() => {
    const updateVoiceStatus = async () => {
      try {
        const response = await fetch(`${API}/api/voice/status`);
        if (!response.ok) return;

        const data = await response.json();
        setVoiceStatus(data);
      } catch (error) {
        console.error("Voice status error:", error);
      }
    };

    updateVoiceStatus();

    const interval = setInterval(updateVoiceStatus, 500);

    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    let mounted = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${API}/api/health`);

        if (!response.ok) {
          throw new Error("Backend unavailable");
        }

        if (mounted) {
          setBackendOnline(true);
        }
      } catch {
        if (mounted) {
          setBackendOnline(false);
        }
      }
    };

    checkBackend();

    const interval = setInterval(checkBackend, 3000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // -----------------------------------------
  // LIVE CAMERA STATUS
  // -----------------------------------------

  useEffect(() => {
    let mounted = true;

    const updateCameraStatus = async () => {
      try {
        const response = await fetch(
          `${API}/api/camera/status`
        );

        if (!response.ok) {
          throw new Error("Unable to retrieve camera status");
        }

        const data = await response.json();

        if (mounted) {
          setCameraStatus(data);
          setCameraError("");
        }
      } catch (error) {
        if (mounted && backendOnline) {
          setCameraError(error.message);
        }
      }
    };

    updateCameraStatus();

    const interval = setInterval(
      updateCameraStatus,
      1000
    );

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [backendOnline]);

  // -----------------------------------------
  // START CAMERA
  // -----------------------------------------

  const startCamera = async () => {
    try {
      setLoading(true);
      setCameraError("");

      const response = await fetch(
        `${API}/api/camera/start`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to start camera"
        );
      }

      setStreamVersion((previous) => previous + 1);

    } catch (error) {
      setCameraError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------------------
  // STOP CAMERA
  // -----------------------------------------

  const stopCamera = async () => {
    try {
      setLoading(true);
      setCameraError("");

      const response = await fetch(
        `${API}/api/camera/stop`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Unable to stop camera");
      }

      setCameraStatus((previous) => ({
        ...previous,
        camera: "Stopped",
        model_status: "Stopped",
        warning: false,
        message: "Camera stopped",
      }));

    } catch (error) {
      setCameraError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------------------
  // STATUS HELPERS
  // -----------------------------------------

  const cameraRunning =
    cameraStatus.camera === "Running";

  const modelStatus =
    cameraStatus.model_status || "Warming up";

  const isDrowsy = modelStatus === "Drowsy";

  const isWarning = Boolean(cameraStatus.warning);
  const driverDisplayStatus = isWarning ? "Drowsy" : modelStatus;

  const navigation = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Live Monitoring", icon: Video },
    { name: "Analytics", icon: Activity },
    { name: "Alert History", icon: Bell },
    { name: "Voice Assistant", icon: Mic },
    { name: "System Settings", icon: Settings },
  ];

  // -----------------------------------------
  // HARDWARE STATUS
  // -----------------------------------------

  const services = [
    {
      name: "MediaPipe",
      description: "Facial landmark detection",
      connected: cameraRunning,
      status: cameraRunning ? "Running" : "Not running",
    },
    {
      name: "Random Forest",
      description: "Drowsiness classification",
      connected:
        cameraRunning &&
        modelStatus !== "Warming up" &&
        modelStatus !== "No face detected",
      status:
        modelStatus === "Alert" ||
        modelStatus === "Drowsy"
          ? "Active"
          : "Waiting",
    },
    {
      name: "Arduino",
      description: `Hardware alert controller${arduinoStatus.port ? ` (${arduinoStatus.port})` : ""}`,
      connected: Boolean(arduinoStatus.connected),
      status: arduinoStatus.connected
        ? arduinoStatus.alert_active
          ? "Connected · Alert Active"
          : "Connected · Alert Inactive"
        : "Disconnected",
    },
    {
      name: "Voice Assistant",
      description: "Gemini and speech services",
      connected: voiceStatus.active,
      status: voiceStatus.active ? voiceStatus.status : "Waiting",
    },
  ];

  // -----------------------------------------
  // DASHBOARD
  // -----------------------------------------

  return (
    <div className="app-container">

      {/* SIDEBAR */}

      <aside className="sidebar">

        <div className="brand">
          <div className="brand-icon">
            <CarFront size={27} />
          </div>

          <div>
            <h2>DriveSafe AI</h2>
            <span>DRIVER MONITORING SYSTEM</span>
          </div>
        </div>

        <div className="nav-label">
          MAIN MENU
        </div>

        <nav>
          {navigation.map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={`nav-item ${
                activePage === name ? "active" : ""
              }`}
              onClick={() => setActivePage(name)}
            >
              <Icon size={19} />
              <span>{name}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">

          <div className="system-card">
            <div className="system-icon">
              <ShieldCheck size={22} />
            </div>

            <h4>Safety First</h4>

            <p>
              AI-powered driver monitoring designed
              to identify signs of drowsiness.
            </p>
          </div>

          <div className="profile">
            <div className="profile-avatar">
              DS
            </div>

            <div>
              <h4>Driver Safety</h4>
              <span>Monitoring System</span>
            </div>
          </div>

        </div>
      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">

        {/* TOP HEADER */}

        <header className="topbar">

          <div>
            <p className="breadcrumb">
              DriveSafe AI / {activePage}
            </p>

            <h1>{activePage}</h1>
          </div>

          <div className="topbar-right">

            <div
              className={`connection-status ${
                backendOnline ? "online" : ""
              }`}
            >
              {backendOnline ? (
                <Wifi size={15} />
              ) : (
                <WifiOff size={15} />
              )}

              <span>
                {backendOnline
                  ? "Backend Connected"
                  : "Backend Offline"}
              </span>
            </div>

            <button className="notification-button">
              <Bell size={19} />
            </button>

            <div className="date-label">
              <Clock3 size={16} />
              Live Monitoring
            </div>

          </div>
        </header>

        {activePage === "Analytics" ? (
          <Analytics />
        ) : (
          <>
        {/* WELCOME */}

        <section className="welcome-section">

          <div>
            <h2>Driver Safety Overview</h2>

            <p>
              Monitor driver alertness and detect
              signs of fatigue in real time.
            </p>
          </div>

          <div
            className="monitoring-badge"
            style={
              isWarning
                ? {
                    color: "#ff9292",
                    background: "#401f29",
                  }
                : cameraRunning
                ? {
                    color: "#69d6a1",
                    background: "#18382f",
                  }
                : {}
            }
          >
            <span
              className="status-dot"
              style={
                isWarning
                  ? { background: "#ff6262" }
                  : cameraRunning
                  ? { background: "#69d6a1" }
                  : {}
              }
            />

            {isWarning
              ? "DROWSINESS WARNING"
              : cameraRunning
              ? "MONITORING ACTIVE"
              : "SYSTEM STANDBY"}
          </div>

        </section>

        {/* STATISTICS */}

        <section className="stats-grid">

          <div className="stat-card">

            <div className="stat-top">
              <span>Driver Status</span>

              <div className="stat-icon cyan">
                <Eye size={21} />
              </div>
            </div>

            <h2
              className="neutral-text"
              style={
                isWarning
                  ? { color: "#ff7777" }
                  : isDrowsy
                  ? { color: "#f0b36c" }
                  : {}
              }
            >
              {driverDisplayStatus}
            </h2>

            <p>
              {cameraStatus.face_detected
                ? "Face detected"
                : cameraRunning
                ? "Waiting for face"
                : "Camera not started"}
            </p>

          </div>

          <div className="stat-card">

            <div className="stat-top">
              <span>Detection Model</span>

              <div className="stat-icon purple">
                <Cpu size={21} />
              </div>
            </div>

            <h2>Random Forest</h2>

            <p>
              {modelStatus === "Alert" ||
              modelStatus === "Drowsy"
                ? "Prediction available"
                : "Waiting for prediction"}
            </p>

          </div>

          <div className="stat-card">

            <div className="stat-top">
              <span>Recent Yawns</span>

              <div className="stat-icon orange">
                <Activity size={21} />
              </div>
            </div>

            <h2>
              {cameraStatus.recent_yawns ?? 0} / 3
            </h2>

            <p>60-second detection window</p>

          </div>

          <div className="stat-card">

            <div className="stat-top">
              <span>System Status</span>

              <div className="stat-icon green">
                <Gauge size={21} />
              </div>
            </div>

            <h2 className="neutral-text">
              {cameraRunning
                ? "Active"
                : cameraStatus.camera === "Error"
                ? "Error"
                : "Standby"}
            </h2>

            <p>
              {backendOnline
                ? "Backend available"
                : "Backend unavailable"}
            </p>

          </div>

        </section>

        {/* CAMERA AND METRICS */}

        <section className="monitor-grid">

          {/* CAMERA PANEL */}

          <div className="panel camera-panel">

            <div className="panel-header">

              <div>
                <h3>Live Driver Monitoring</h3>
                <p>Real-time facial observation</p>
              </div>

              <span className="live-tag">
                <span className="status-dot" />
                {cameraRunning ? "LIVE" : "CAMERA"}
              </span>

            </div>

            <div className="camera-placeholder">

              {cameraRunning ? (

                <img
                  key={streamVersion}
                  src={`${API}/api/camera/video?v=${streamVersion}`}
                  alt="Live driver monitoring camera"
                  className="live-camera-feed"
                />

              ) : (

                <div className="camera-content">

                  <div className="camera-icon">
                    <Camera size={38} />
                  </div>

                  <h3>
                    {cameraStatus.camera === "Error"
                      ? "Camera Error"
                      : "Camera Feed Unavailable"}
                  </h3>

                  <p>
                    {cameraError ||
                      cameraStatus.message ||
                      "Start the camera to begin monitoring."}
                  </p>

                  <span className="camera-note">
                    {cameraStatus.camera.toUpperCase()}
                  </span>

                </div>

              )}

            </div>

            <div className="camera-controls">

              <button
                className="camera-start-button"
                onClick={startCamera}
                disabled={
                  !backendOnline ||
                  cameraRunning ||
                  loading
                }
              >
                <Camera size={16} />

                {loading
                  ? "Please wait..."
                  : "Start Monitoring"}
              </button>

              <button
                className="camera-stop-button"
                onClick={stopCamera}
                disabled={
                  !cameraRunning ||
                  loading
                }
              >
                <Square size={14} />
                Stop Camera
              </button>

            </div>

            {cameraError && (
              <p
                style={{
                  color: "#ff8585",
                  fontSize: "11px",
                  marginTop: "12px",
                }}
              >
                {cameraError}
              </p>
            )}

            <div className="voice-status-panel">
              <div className="voice-status-icon">
                <Mic size={20} />
              </div>

              <div>
                <h4>
                  {voiceStatus.status === "Listening"
                    ? "🎤 Hearing you..."
                    : voiceStatus.status === "Speaking"
                    ? "🔊 Speaking..."
                    : voiceStatus.status === "Thinking"
                    ? "🧠 Thinking..."
                    : "Voice Assistant"}
                </h4>
                <p>{voiceStatus.message}</p>
              </div>

              <span
                className={`voice-indicator ${
                  voiceStatus.active ? "voice-active" : ""
                }`}
              />
            </div>

            <div className="camera-footer">

              <span>
                {cameraRunning ? (
                  <>
                    <Wifi size={15} />
                    Camera connected
                  </>
                ) : (
                  <>
                    <WifiOff size={15} />
                    Camera disconnected
                  </>
                )}
              </span>

              <span>
                MediaPipe Face Landmarker
              </span>

            </div>

          </div>

          {/* DETECTION METRICS */}

          <div className="panel detection-panel">

            <div className="panel-header">

              <div>
                <h3>Detection Metrics</h3>
                <p>Facial feature measurements</p>
              </div>

              <Activity
                size={19}
                className="muted-icon"
              />

            </div>

            <div className="metric-box">

              <div>
                <span>Eye Aspect Ratio</span>
                <p>EAR</p>
              </div>

              <strong>
                {Number.isFinite(cameraStatus.ear)
                  ? cameraStatus.ear.toFixed(3)
                  : "--"}
              </strong>

            </div>

            <div className="metric-box">

              <div>
                <span>Mouth Aspect Ratio</span>
                <p>MAR</p>
              </div>

              <strong>
                {Number.isFinite(cameraStatus.mar)
                  ? cameraStatus.mar.toFixed(3)
                  : "--"}
              </strong>

            </div>

            <div className="metric-box">

              <div>
                <span>Model Prediction</span>
                <p>Random Forest</p>
              </div>

              <strong className="waiting-label">
                {modelStatus}
              </strong>

            </div>

          </div>

        </section>

        {/* ALERTS AND SERVICES */}

        <section className="bottom-grid">

          {/* ALERT PANEL */}

          <div className="panel alert-panel">

            <div className="panel-header">

              <div>
                <h3>Alert Management</h3>
                <p>Driver safety notifications</p>
              </div>

              <AlertTriangle
                size={20}
                className="muted-icon"
              />

            </div>

            {isWarning ? (

              <div
                className="empty-state"
                style={{
                  background: "#351e29",
                  color: "#ff7777",
                }}
              >

                <AlertTriangle size={27} />

                <div>
                  <h4>DROWSINESS ALERT</h4>

                  <p>
                    {cameraStatus.message ||
                      "Please pull over safely and rest."}
                  </p>
                </div>

              </div>

            ) : (

              <div className="empty-state">

                <CheckCircle2 size={27} />

                <div>
                  <h4>No active alerts</h4>

                  <p>
                    {cameraRunning
                      ? "Driver monitoring is active."
                      : "Start monitoring to detect alerts."}
                  </p>
                </div>

              </div>

            )}

          </div>

          {/* HARDWARE STATUS */}

          <div className="panel hardware-panel">

            <div className="panel-header">

              <div>
                <h3>Hardware & Services</h3>
                <p>Connected components</p>
              </div>

              <Zap
                size={20}
                className="muted-icon"
              />

            </div>

            {services.map((service) => (

              <div
                className="service-row"
                key={service.name}
              >

                <div>

                  <span
                    className="service-dot"
                    style={{
                      background: service.connected
                        ? "#69d6a1"
                        : "#dba96d",
                    }}
                  />

                  <div>
                    <h4>{service.name}</h4>
                    <p>{service.description}</p>
                  </div>

                </div>

                <span
                  className="offline-label"
                  style={
                    service.connected
                      ? {
                          color: "#69d6a1",
                          background: "#18382f",
                        }
                      : {}
                  }
                >
                  {service.status}
                </span>

              </div>

            ))}

          </div>

        </section>

        {/* FOOTER */}

        <footer className="footer">

          <span>
            DriveSafe AI · Driver Drowsiness Detection System
          </span>

          <span>
            AI-powered driver safety monitoring
          </span>

        </footer>
          </>
        )}

      </main>

    </div>
  );
}

export default App;
