// ===== Toggle Doctor/Patient Fields =====
const userTypeRadios = document.querySelectorAll('input[name="user_type"]');
const doctorFields = document.getElementById('doctor-fields');
const patientFields = document.getElementById('patient-fields');

if (userTypeRadios.length > 0) {
    userTypeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === 'doctor') {
                doctorFields?.classList.remove('hidden');
                patientFields?.classList.add('hidden');
            } else {
                doctorFields?.classList.add('hidden');
                patientFields?.classList.remove('hidden');
            }
        });
    });
}

// ===== Handle Doctor Registration =====
function handleDoctorRegister(event) {
    event.preventDefault();

    const fullname = document.getElementById('fullname').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    const phone = document.getElementById('phone').value.trim();
    const license = document.getElementById('license').value.trim();
    const specialization = document.getElementById('specialization').value.trim();
    const hospital = document.getElementById('hospital').value.trim();

    if (!fullname) {
        showAlert('Please enter full name', 'error');
        return;
    }

    if (!isValidEmail(email)) {
        showAlert('Invalid email format', 'error');
        return;
    }

    if (password.length < 8) {
        showAlert('Password must be at least 8 characters', 'error');
        return;
    }

    if (password !== confirmPassword) {
        showAlert('Passwords do not match', 'error');
        return;
    }

    if (!isValidPhone(phone)) {
        showAlert('Phone must be 10 digits', 'error');
        return;
    }

    if (!license) {
        showAlert('Please enter medical license number', 'error');
        return;
    }

    if (!specialization) {
        showAlert('Please enter specialization', 'error');
        return;
    }

    if (!hospital) {
        showAlert('Please enter hospital/clinic name', 'error');
        return;
    }

    fetch('/api/doctor-register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({fullname, email, password, phone, license, specialization, hospital})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showAlert('Doctor registered! Redirecting to login...', 'success');
            setTimeout(() => { window.location.href = '/login.html'; }, 1500);
        } else {
            showAlert(data.message, 'error');
        }
    })
    .catch(e => showAlert('Error: ' + e.message, 'error'));
}

// ===== Handle Patient Registration =====
function handlePatientRegister(event) {
    event.preventDefault();

    const fullname = document.getElementById('fullname').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    const phone = document.getElementById('phone').value.trim();
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const medicalHistory = document.getElementById('medical-history').value.trim();

    if (!fullname) {
        showAlert('Please enter full name', 'error');
        return;
    }

    if (!isValidEmail(email)) {
        showAlert('Invalid email format', 'error');
        return;
    }

    if (password.length < 8) {
        showAlert('Password must be at least 8 characters', 'error');
        return;
    }

    if (password !== confirmPassword) {
        showAlert('Passwords do not match', 'error');
        return;
    }

    if (!isValidPhone(phone)) {
        showAlert('Phone must be 10 digits', 'error');
        return;
    }

    if (!age || age < 1 || age > 120) {
        showAlert('Please enter valid age', 'error');
        return;
    }

    if (!gender) {
        showAlert('Please select gender', 'error');
        return;
    }

    fetch('/api/patient-register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({fullname, email, password, phone, age, gender, medical_history: medicalHistory})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showAlert('Patient registered! Redirecting to login...', 'success');
            setTimeout(() => { window.location.href = '/login.html'; }, 1500);
        } else {
            showAlert(data.message, 'error');
        }
    })
    .catch(e => showAlert('Error: ' + e.message, 'error'));
}

// ===== Handle Doctor Login =====
function handleDoctorLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('doctor-email').value.trim();
    const password = document.getElementById('doctor-password').value;

    if (!email) {
        showAlert('Please enter email', 'error');
        return;
    }

    if (!isValidEmail(email)) {
        showAlert('Invalid email format', 'error');
        return;
    }

    if (!password) {
        showAlert('Please enter password', 'error');
        return;
    }

    fetch('/api/doctor-login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({email, password})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showAlert('Login successful! Redirecting...', 'success');
            setTimeout(() => { window.location.href = '/doctor_dashboard.html'; }, 1500);
        } else {
            showAlert(data.message, 'error');
        }
    })
    .catch(e => showAlert('Error: ' + e.message, 'error'));
}

// ===== Handle Patient Login =====
function handlePatientLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('patient-email').value.trim();
    const password = document.getElementById('patient-password').value;

    if (!email) {
        showAlert('Please enter email', 'error');
        return;
    }

    if (!isValidEmail(email)) {
        showAlert('Invalid email format', 'error');
        return;
    }

    if (!password) {
        showAlert('Please enter password', 'error');
        return;
    }

    fetch('/api/patient-login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({email, password})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showAlert('Login successful! Redirecting...', 'success');
            setTimeout(() => { window.location.href = '/patient_dashboard.html'; }, 1500);
        } else {
            showAlert(data.message, 'error');
        }
    })
    .catch(e => showAlert('Error: ' + e.message, 'error'));
}

// ===== Email Validation =====
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// ===== Phone Validation =====
function isValidPhone(phone) {
    const phoneRegex = /^[0-9]{10}$/;
    return phoneRegex.test(phone.replace(/[\s\-\(\)]/g, ''));
}

// ===== Show Alert =====
function showAlert(message, type = 'info') {
    const existingAlert = document.querySelector('.alert');
    if (existingAlert) existingAlert.remove();

    const alert = document.createElement('div');
    alert.className = 'alert';
    alert.textContent = message;

    let bgColor = '#004d40';
    if (type === 'error') bgColor = '#d32f2f';
    if (type === 'success') bgColor = '#388e3c';

    alert.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background-color: ${bgColor};
        color: white;
        border-radius: 5px;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(alert);
    setTimeout(() => alert.remove(), 3000);
}

// ===== Smooth Scroll =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        const target = document.querySelector(href);
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// ===== Page Load Animations =====
window.addEventListener('load', function() {
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach((card, i) => {
        card.style.animation = `fadeInUp 0.6s ease ${i * 0.1}s forwards`;
    });
});

// ===== Animations =====
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);

// ===== Form Input Focus =====
document.querySelectorAll('input, select, textarea').forEach(input => {
    input.addEventListener('focus', function() {
        this.style.transition = 'all 0.3s ease';
    });
});

function handleMRIUpload(event) {
    event.preventDefault();
    
    const patientEmail = document.getElementById('patient-email').value.trim();
    const mriFile = document.getElementById('mri-file').files[0];
    
    if (!patientEmail) {
        showAlert('Please enter patient email', 'error');
        return;
    }
    
    if (!mriFile) {
        showAlert('Please select MRI file', 'error');
        return;
    }
    
    // ✅ Create FormData before using it
    const formData = new FormData();
    formData.append('patient_email', patientEmail);
    formData.append('mri_file', mriFile);
    
    showAlert('Analyzing MRI scan...', 'info');
    
    fetch('/api/predict-mri', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            displayPredictionResult(data);
            showAlert('Analysis complete!', 'success');
            document.getElementById('mri-file').value = '';
            document.getElementById('patient-email').value = '';
        } else {
            showAlert(data.message, 'error');
        }
    })
    .catch(e => showAlert('Error: ' + e.message, 'error'));
}

// Display prediction results
function displayPredictionResult(data) {
    const resultDiv = document.getElementById('prediction-result');
    document.getElementById('pred-class').textContent = data.prediction;
    document.getElementById('pred-confidence').textContent = data.confidence + '%';
    
    // Class probabilities
    const classesHtml = Object.entries(data.classes)
        .map(([className, prob]) => `
            <div style="margin-bottom: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>${className}</span>
                    <span style="font-weight: 600;">${(prob * 100).toFixed(2)}%</span>
                </div>
                <div style="background: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: #004d40; height: 100%; width: ${prob * 100}%;"></div>
                </div>
            </div>
        `).join('');
    
    document.getElementById('pred-classes').innerHTML = classesHtml;
    document.getElementById('pred-suggestions').textContent = data.suggestions;
    
    resultDiv.style.display = 'block';
}
