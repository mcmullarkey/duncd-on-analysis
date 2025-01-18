let chart = null;

async function findGoodEpisodes() {
    const aboutText = document.querySelector('.about-text');
    const contactText = document.querySelector('.contact-text');
    const chartContainer = document.getElementById('chart-container');
    
    // Fade out about text and fade in contact text
    aboutText.style.opacity = '0';
    setTimeout(() => {
        contactText.style.opacity = '1';
    }, 500);
    
    try {
        const response = await fetch('/predict');
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Show and fade in the chart container
        chartContainer.classList.add('visible');
        
        // Slight delay before updating chart for smooth transition
        setTimeout(() => {
            updateChart(data);
        }, 100);
        
    } catch (error) {
        alert('Error fetching predictions');
        console.error(error);
    }
}

function updateChart(data) {
    if (chart) {
        chart.destroy();
    }

    const ctx = document.getElementById('episodeChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.episode),
            datasets: [{
                label: 'Banger Probability',
                data: data.map(item => item.probability),
                backgroundColor: 'rgba(98, 0, 234, 0.8)',
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Probability: ${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Probability of being a banger',
                        color: '#fff'
                    },
                    ticks: {
                        color: '#fff'
                    }
                },
                y: {
                    ticks: {
                        color: '#fff'
                    }
                }
            }
        }
    });
}