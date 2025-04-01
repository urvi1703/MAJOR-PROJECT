## Prerequisites (For Windows Users)
Before installing dependencies, install **PortAudio** manually:

1. Download **PortAudio binaries** from [this link](http://www.portaudio.com/download.html).
2. Extract the files and copy `portaudio_x64.dll` to `C:\Windows\System32`.
3. Restart your system to apply changes.

### Install Python Dependencies:
```bash
pip install -r requirements.txt

---

### **3. Commit & Push Changes to GitHub**
Run the following in **Git Bash** or **Command Prompt**:  
```bash
git add requirements.txt README.md
git commit -m "Updated dependencies and Windows installation guide"
git push origin main
