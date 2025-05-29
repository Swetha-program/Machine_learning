document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const form = document.getElementById('strokeForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            let isValid = true;
            
            // Validate all required fields
            const inputs = form.querySelectorAll('input[required], select[required]');
            inputs.forEach(input => {
                if (!input.value) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            // Validate numeric fields
            const numericFields = ['age', 'avg_glucose_level', 'bmi'];
            numericFields.forEach(fieldId => {
                const field = document.getElementById(fieldId);
                if (field && isNaN(field.value)) {
                    isValid = false;
                    field.classList.add('is-invalid');
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields with valid values.');
            }
        });
    }
    
    // Add Bootstrap validation classes
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.checkValidity()) {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            } else {
                this.classList.remove('is-valid');
            }
        });
    });
});