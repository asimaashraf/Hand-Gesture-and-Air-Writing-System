class BrowserCamera {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser does not support camera access.");
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    this.video.srcObject = this.stream;
    await this.video.play();
    return {
      width: this.video.videoWidth || 1280,
      height: this.video.videoHeight || 720
    };
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.video.pause();
    this.video.srcObject = null;
  }

  get active() {
    return Boolean(this.stream);
  }
}

window.BrowserCamera = BrowserCamera;
