class BrowserOCR {
  constructor(outputElement) {
    this.output = outputElement;
    this.worker = null;
  }

  async recognize(canvas) {
    if (!window.Tesseract) {
      throw new Error("Tesseract.js did not load.");
    }

    this.output.value = "Recognizing text...";
    const image = this.prepareCanvas(canvas);
    const result = await Tesseract.recognize(image, "eng", {
      logger: (message) => {
        if (message.status) {
          const progress = message.progress ? ` ${Math.round(message.progress * 100)}%` : "";
          this.output.value = `${message.status}${progress}`;
        }
      }
    });

    const text = result.data.text.trim();
    this.output.value = text || "No text detected.";
    return this.output.value;
  }

  prepareCanvas(source) {
    const out = document.createElement("canvas");
    out.width = source.width;
    out.height = source.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(source, 0, 0);

    const image = ctx.getImageData(0, 0, out.width, out.height);
    for (let i = 0; i < image.data.length; i += 4) {
      const r = image.data[i];
      const g = image.data[i + 1];
      const b = image.data[i + 2];
      const luminance = (r * 0.299) + (g * 0.587) + (b * 0.114);
      const value = luminance < 245 ? 0 : 255;
      image.data[i] = value;
      image.data[i + 1] = value;
      image.data[i + 2] = value;
      image.data[i + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return out;
  }
}

window.BrowserOCR = BrowserOCR;
