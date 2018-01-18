// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

function toggleHelpBox() {
  var helpBoxOuter = document.getElementById('details');
  helpBoxOuter.classList.toggle(HIDDEN_CLASS);
  var detailsButton = document.getElementById('details-button');
  if (helpBoxOuter.classList.contains(HIDDEN_CLASS))
    detailsButton.innerText = detailsButton.detailsText;
  else
    detailsButton.innerText = detailsButton.hideDetailsText;

  // Details appears over the main content on small screens.
  if (mobileNav) {
    document.getElementById('main-content').classList.toggle(HIDDEN_CLASS);
    var runnerContainer = document.querySelector('.runner-container');
    if (runnerContainer) {
      runnerContainer.classList.toggle(HIDDEN_CLASS);
    }
  }
}

function diagnoseErrors() {
// <if expr="not chromeos">
    if (window.errorPageController)
      errorPageController.diagnoseErrorsButtonClick();
// </if>
// <if expr="chromeos">
  var extensionId = 'idddmepepmjcgiedknnmlbadcokidhoa';
  var diagnoseFrame = document.getElementById('diagnose-frame');
  diagnoseFrame.innerHTML =
      '<iframe src="chrome-extension://' + extensionId +
      '/index.html"></iframe>';
// </if>
}

// Subframes use a different layout but the same html file.  This is to make it
// easier to support platforms that load the error page via different
// mechanisms (Currently just iOS).
if (window.top.location != window.location)
  document.documentElement.setAttribute('subframe', '');

// Re-renders the error page using |strings| as the dictionary of values.
// Used by NetErrorTabHelper to update DNS error pages with probe results.
function updateForDnsProbe(strings) {
  var context = new JsEvalContext(strings);
  jstProcess(context, document.getElementById('t'));
}

// Given the classList property of an element, adds an icon class to the list
// and removes the previously-
function updateIconClass(classList, newClass) {
  var oldClass;

  if (classList.hasOwnProperty('last_icon_class')) {
    oldClass = classList['last_icon_class'];
    if (oldClass == newClass)
      return;
  }

  classList.add(newClass);
  if (oldClass !== undefined)
    classList.remove(oldClass);

  classList['last_icon_class'] = newClass;

  if (newClass == 'icon-offline') {
    document.body.classList.add('offline');
    window.runner = new Runner('.interstitial-wrapper');
  } else {
    document.body.classList.add('neterror');
  }
}

var primaryControlOnLeft = true;
// <if expr="is_macosx or is_ios or is_linux or is_android">
primaryControlOnLeft = false;
// </if>

function onDocumentLoad() {
}

document.addEventListener('DOMContentLoaded', onDocumentLoad);
