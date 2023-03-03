// Add alerts to the top of the page, if any are present
$(document).ready(function () {
  $.get(ROOT_URL + 'alert.html', function (data) {
      if (data.length > 0) {
        const content = $('<div class="news-alert">' + data + '</div>')
        $('#banner').prepend(content);
      }
  })
})


// Display a version warning banner if necessary
$(document).ready(function () {
    $.getJSON(ROOT_URL + 'versions.json' , function (data) {
      if (CURRENT_VERSION != data.latest) {
        let msg
        if (data.all.indexOf(CURRENT_VERSION) < 0 ) {
          msg = "DEVELOPMENT / PRE-RELEASE"
        } else {
          msg = "PREVIOUS RELEASE"
        }
        const content = $('<div class="version-alert">This page is documentation for a ' + msg + ' version. For the latest release, click <a href="' + ROOT_URL + data.latest + '">here</a></div>')
        $('#banner').append(content);
      }
      for (let i = 0; i < data.menu.length; i++) {
        const link = ROOT_URL + data.menu[i];
        $('#version-menu').append('<a class="dropdown-item" href="' + link + '">' + data.menu[i] + '</a>');
      }
    })
    .fail(function() {
        // if fetching version file fails, create a one line dropdown version picker
        $('#version-menu').append('<a class="dropdown-item" href="#">' + CURRENT_VERSION + '</a>');
    });
})
