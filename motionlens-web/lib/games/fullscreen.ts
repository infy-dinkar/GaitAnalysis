// Fullscreen helpers.
//
// Browsers only honour requestFullscreen() inside a user gesture, so
// the call has to ride on a click the patient already makes — in this
// game, choosing a hand. It cannot be fired when a countdown ends.
//
// Every function here swallows its errors. Fullscreen is a nicety; a
// browser or a policy that refuses it must never stop the game.

type FsElement = HTMLElement & {
  webkitRequestFullscreen?: () => Promise<void> | void;
  msRequestFullscreen?: () => Promise<void> | void;
};

type FsDocument = Document & {
  webkitExitFullscreen?: () => Promise<void> | void;
  msExitFullscreen?: () => Promise<void> | void;
  webkitFullscreenElement?: Element | null;
  msFullscreenElement?: Element | null;
};

/** True when anything is currently fullscreen. */
export function isFullscreen(): boolean {
  if (typeof document === "undefined") return false;
  const d = document as FsDocument;
  return !!(d.fullscreenElement ?? d.webkitFullscreenElement ?? d.msFullscreenElement);
}

/**
 * Ask for fullscreen on `el`.
 *
 * MUST be called synchronously from a user gesture handler.
 * @returns true if the request was accepted.
 */
export async function enterFullscreen(el: HTMLElement | null): Promise<boolean> {
  if (!el) return false;
  const e = el as FsElement;
  const req =
    e.requestFullscreen?.bind(e)
    ?? e.webkitRequestFullscreen?.bind(e)
    ?? e.msRequestFullscreen?.bind(e);
  if (!req) return false;
  try {
    await req();
    return true;
  } catch {
    // Refused by the browser, an iframe policy, or the user. Carry on.
    return false;
  }
}

export async function exitFullscreen(): Promise<void> {
  if (typeof document === "undefined") return;
  if (!isFullscreen()) return;
  const d = document as FsDocument;
  const exit =
    d.exitFullscreen?.bind(d)
    ?? d.webkitExitFullscreen?.bind(d)
    ?? d.msExitFullscreen?.bind(d);
  if (!exit) return;
  try {
    await exit();
  } catch {
    // Nothing useful to do; leaving the page will clear it anyway.
  }
}

/** Subscribe to fullscreen enter/exit, including Esc. Returns an
 *  unsubscribe function. */
export function onFullscreenChange(cb: () => void): () => void {
  if (typeof document === "undefined") return () => {};
  document.addEventListener("fullscreenchange", cb);
  document.addEventListener("webkitfullscreenchange", cb);
  return () => {
    document.removeEventListener("fullscreenchange", cb);
    document.removeEventListener("webkitfullscreenchange", cb);
  };
}
