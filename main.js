const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

let mainWindow;
let streamlitProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        icon: path.join(__dirname, 'assets', 'icon.png'),
        title: 'VisualAI Studio'
    });

    // Load aplikasi Streamlit
    mainWindow.loadURL('http://localhost:8501');
    
    // Open DevTools jika diperlukan
    // mainWindow.webContents.openDevTools();

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function startStreamlit() {
    const python = process.platform === 'win32' ? 'python' : 'python3';
    
    streamlitProcess = spawn(python, ['-m', 'streamlit', 'run', 'app.py', '--server.port=8501', '--server.headless=true']);
    
    streamlitProcess.stdout.on('data', (data) => {
        console.log(`Streamlit: ${data}`);
    });
    
    streamlitProcess.stderr.on('data', (data) => {
        console.error(`Streamlit Error: ${data}`);
    });
    
    streamlitProcess.on('close', (code) => {
        console.log(`Streamlit process exited with code ${code}`);
    });
}

function waitForStreamlit() {
    const checkServer = () => {
        http.get('http://localhost:8501', (res) => {
            if (res.statusCode === 200) {
                createWindow();
            } else {
                setTimeout(checkServer, 1000);
            }
        }).on('error', () => {
            setTimeout(checkServer, 1000);
        });
    };
    
    checkServer();
}

app.whenReady().then(() => {
    startStreamlit();
    setTimeout(waitForStreamlit, 2000);
});

app.on('window-all-closed', () => {
    if (streamlitProcess) {
        streamlitProcess.kill();
    }
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});