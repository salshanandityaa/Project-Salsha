/*!
    * Start Bootstrap - SB Admin (v7.0.5) - Dapatkan dari template asli Anda
    * Copyright 2013-2021 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-sb-admin/blob/master/LICENSE)
    */
//
// Scripts inti dari template SB Admin 2
//

window.addEventListener('DOMContentLoaded', event => {

    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        // Event listener untuk menyimpan state sidebar saat di-toggle
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }

    // Collapse the sidebar if desktop and not in toggled state
    if (window.innerWidth > 768 && !document.body.classList.contains('sb-sidenav-toggled')) {
        if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
             document.body.classList.toggle('sb-sidenav-toggled');
        }
    }
    
    // Activate feather icons if used (hapus baris ini jika 'feather is not defined' error muncul dan Anda tidak menggunakannya)
    // feather.replace(); 

    // Menambahkan fungsi collapse pada sidebar accordion items
    const accordionSidenav = document.body.querySelector('#sidenavAccordion');
    if (accordionSidenav) {
        new bootstrap.Collapse(accordionSidenav, {
            toggle: false
        });
    }

    // --- KODE KUSTOM ANDA DIMULAI DI SINI ---

    // Logika untuk Menutup Flash Messages Secara Otomatis
    document.querySelectorAll('.alert-dismissible').forEach(function(flashMessage) {
        setTimeout(function() {
            var bsAlert = bootstrap.Alert.getInstance(flashMessage);
            if (bsAlert) {
                bsAlert.close();
            } else {
                new bootstrap.Alert(flashMessage).close();
            }
        }, 5000); // 5000 milidetik = 5 detik
    });

    // Inisialisasi Simple-DataTables
    // Periksa apakah simpleDatatables object ada sebelum menggunakannya
    if (typeof simpleDatatables !== 'undefined') { 
        const usersTable = document.getElementById('usersTable');
        if (usersTable) {
            new simpleDatatables.DataTable(usersTable);
        }
        const reportsDataTable = document.getElementById('reportsDataTable');
        if (reportsDataTable) {
            new simpleDatatables.DataTable(reportsDataTable);
        }
    } else {
        console.warn("Simple-Datatables object not found. DataTables initialization skipped.");
    }

    console.log("scripts.js is loaded and running!"); // Log konfirmasi pemuatan script
});