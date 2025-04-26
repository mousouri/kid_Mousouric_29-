class GradientEffect {
    constructor(options = {}) {
        this.options = {
            colors: options.colors || ['#1a2a6c', '#b21f1f', '#fdbb2d'],
            speed: options.speed || 0.5,
            size: options.size || 150
        };
        
        this.init();
    }

    init() {
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.style.position = 'fixed';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.zIndex = '0';
        document.getElementById('gradient-container').appendChild(this.canvas);
        
        // Get context
        this.ctx = this.canvas.getContext('2d');
        
        // Set size
        this.resize();
        window.addEventListener('resize', this.resize.bind(this));
        
        // Start animation
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    animate() {
        const time = Date.now() * this.options.speed * 0.001;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Create gradient
        const gradient = this.ctx.createRadialGradient(
            this.canvas.width / 2 + Math.cos(time) * this.options.size,
            this.canvas.height / 2 + Math.sin(time) * this.options.size,
            0,
            this.canvas.width / 2,
            this.canvas.height / 2,
            this.options.size * 2
        );
        
        // Add color stops
        this.options.colors.forEach((color, index) => {
            const offset = (index / (this.options.colors.length - 1) + Math.sin(time * 0.5) * 0.1) % 1;
            gradient.addColorStop(offset, color);
        });
        
        // Fill with gradient
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Add subtle noise
        this.addNoise();
        
        requestAnimationFrame(this.animate.bind(this));
    }

    addNoise() {
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            const noise = Math.random() * 10;
            data[i] = Math.min(255, data[i] + noise);
            data[i + 1] = Math.min(255, data[i + 1] + noise);
            data[i + 2] = Math.min(255, data[i + 2] + noise);
        }
        
        this.ctx.putImageData(imageData, 0, 0);
    }

    destroy() {
        window.removeEventListener('resize', this.resize);
        if (this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
} 