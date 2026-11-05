// Adds a "Create account" link to Chainlit's login page.
(function () {
  function addLink() {
    if (!location.pathname.endsWith("/login")) return;
    if (document.getElementById("signup-link")) return;
    var form = document.querySelector("form");
    if (!form) return;
    var p = document.createElement("p");
    p.id = "signup-link";
    p.style.cssText = "text-align:center;margin-top:16px;font-size:14px;";
    p.innerHTML = 'New here? <a href="/signup" style="color:#ff3d8b;font-weight:600;text-decoration:underline;">Create an account</a>';
    form.parentNode.insertBefore(p, form.nextSibling);
  }
  new MutationObserver(addLink).observe(document.documentElement, { childList: true, subtree: true });
  addLink();
})();