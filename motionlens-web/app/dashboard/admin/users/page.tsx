"use client";
// /dashboard/admin/users — admin-only user management.
//
// ⚠️ The isAdmin check below is COSMETIC. It hides the UI and redirects,
// but the JS bundle still ships and anyone can call the API directly.
// The real gate is require_admin on /api/admin/*, which returns 403 —
// every fetch on this page is authorised server-side independently.

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, RefreshCw, ShieldAlert, UserPlus } from "lucide-react";
import { AuthGuard } from "@/components/auth/AuthGuard";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/contexts/AuthContext";
import {
  adminCreateUser,
  adminListUsers,
  adminResetPassword,
  adminUpdateUser,
} from "@/lib/admin";
import type { DoctorPublicDTO } from "@/lib/auth";

const ROLES = ["clinician", "admin"] as const;

export default function AdminUsersPage() {
  return (
    <AuthGuard>
      <DashboardShell backHref="/dashboard" backLabel="Dashboard" title="Users">
        <AdminUsersInner />
      </DashboardShell>
    </AuthGuard>
  );
}

function AdminUsersInner() {
  const { doctor, isAdmin, loading: authLoading } = useAuth();
  const router = useRouter();

  const [users, setUsers] = useState<DoctorPublicDTO[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [resetFor, setResetFor] = useState<DoctorPublicDTO | null>(null);

  // Non-admins get bounced. Waits for authLoading so a hard refresh
  // doesn't redirect before the role has been re-fetched from /me.
  useEffect(() => {
    if (!authLoading && !isAdmin) router.replace("/dashboard");
  }, [authLoading, isAdmin, router]);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await adminListUsers();
      setUsers(res.data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load users");
      setUsers([]);
    }
  }, []);

  // Initial load. Written as .then + a cancelled flag (rather than
  // awaiting `load`) to match the other dashboard pages and to keep the
  // state update out of the effect body itself.
  useEffect(() => {
    if (!isAdmin) return;
    let cancelled = false;
    adminListUsers()
      .then((res) => {
        if (!cancelled) setUsers(res.data);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Could not load users");
        setUsers([]);
      });
    return () => {
      cancelled = true;
    };
  }, [isAdmin]);

  async function patch(u: DoctorPublicDTO, body: { role?: string; is_active?: boolean }) {
    setBusyId(u.id);
    setError(null);
    try {
      const updated = await adminUpdateUser(u.id, body);
      setUsers((prev) =>
        (prev ?? []).map((x) => (x.id === updated.id ? updated : x)),
      );
    } catch (e) {
      // Guard rejections (last active admin, self-deactivation) land
      // here with the backend's own wording.
      setError(e instanceof Error ? e.message : "Update failed");
    } finally {
      setBusyId(null);
    }
  }

  if (authLoading || !isAdmin) {
    return (
      <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 text-muted">
        {authLoading ? (
          <Loader2 className="h-6 w-6 animate-spin text-accent" />
        ) : (
          <>
            <ShieldAlert className="h-8 w-8 text-error" />
            <p className="text-sm">Not authorized — redirecting…</p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Administration</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
            Users
          </h1>
          <p className="mt-2 text-sm text-muted">
            Create clinician and admin accounts, change roles, and reset
            passwords. Patient data stays private to each clinician — admins
            manage accounts only.
          </p>
        </div>
        <Button variant="secondary" onClick={() => void load()}>
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="rounded-card border border-error/40 bg-error/5 p-4 text-sm text-error">
          {error}
        </div>
      )}

      <CreateUserForm
        onCreated={(u) => setUsers((prev) => [u, ...(prev ?? [])])}
        onError={setError}
      />

      <section>
        <h2 className="text-lg font-semibold tracking-tight">
          All users{" "}
          <span className="text-sm font-normal text-muted">
            ({users?.length ?? 0})
          </span>
        </h2>

        {users === null ? (
          <div className="mt-4 flex items-center justify-center py-10">
            <Loader2 className="h-6 w-6 animate-spin text-accent" />
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto rounded-card border border-border">
            <table className="w-full min-w-[52rem] text-sm">
              <thead>
                <tr className="border-b border-border bg-surface/60 text-left">
                  <Th>Name</Th>
                  <Th>Email</Th>
                  <Th>Role</Th>
                  <Th>Active</Th>
                  <Th>Created</Th>
                  <Th>Password</Th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => {
                  const isSelf = u.id === doctor?.id;
                  const busy = busyId === u.id;
                  return (
                    <tr key={u.id} className="border-b border-border last:border-0">
                      <Td>
                        {u.name}
                        {isSelf && (
                          <span className="ml-2 rounded-full bg-accent/15 px-2 py-0.5 text-[10px] uppercase tracking-wide text-accent">
                            you
                          </span>
                        )}
                      </Td>
                      <Td className="text-muted">{u.email}</Td>
                      <Td>
                        <select
                          value={u.role ?? "clinician"}
                          disabled={busy}
                          onChange={(e) => void patch(u, { role: e.target.value })}
                          className="rounded-md border border-border bg-surface px-2 py-1 text-sm"
                        >
                          {ROLES.map((r) => (
                            <option key={r} value={r}>
                              {r}
                            </option>
                          ))}
                        </select>
                      </Td>
                      <Td>
                        <Button
                          variant="secondary"
                          size="sm"
                          disabled={busy || isSelf}
                          // Own row disabled: the backend rejects
                          // self-deactivation anyway (400), this just
                          // avoids offering the click.
                          title={
                            isSelf ? "You cannot deactivate your own account" : ""
                          }
                          onClick={() =>
                            void patch(u, { is_active: !(u.is_active ?? true) })
                          }
                          className={
                            u.is_active ?? true
                              ? "text-emerald-500"
                              : "text-error"
                          }
                        >
                          {u.is_active ?? true ? "Active" : "Inactive"}
                        </Button>
                      </Td>
                      <Td className="text-muted">
                        {new Date(u.created_at).toLocaleDateString()}
                      </Td>
                      <Td>
                        <Button
                          variant="secondary"
                          size="sm"
                          disabled={busy}
                          onClick={() => setResetFor(u)}
                        >
                          Reset
                        </Button>
                      </Td>
                    </tr>
                  );
                })}
                {users.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-muted">
                      No users found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {resetFor && (
        <ResetPasswordDialog
          user={resetFor}
          onClose={() => setResetFor(null)}
          onError={setError}
        />
      )}
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-4 py-3 text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
      {children}
    </th>
  );
}

function Td({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <td className={`px-4 py-3 align-middle ${className}`}>{children}</td>;
}

// ─── Create user ─────────────────────────────────────────────────
function CreateUserForm({
  onCreated,
  onError,
}: {
  onCreated: (u: DoctorPublicDTO) => void;
  onError: (msg: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<string>("clinician");
  const [specialization, setSpecialization] = useState("");
  const [license, setLicense] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    onError(null);
    try {
      const created = await adminCreateUser({
        name: name.trim(),
        email: email.trim(),
        password,
        role,
        specialization: specialization.trim() || null,
        license_number: license.trim() || null,
      });
      onCreated(created);
      setName("");
      setEmail("");
      setPassword("");
      setRole("clinician");
      setSpecialization("");
      setLicense("");
      setOpen(false);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not create user");
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <Button onClick={() => setOpen(true)}>
        <UserPlus className="h-4 w-4" />
        Add user
      </Button>
    );
  }

  return (
    <form
      onSubmit={submit}
      className="rounded-card border border-border bg-surface p-5 md:p-6"
    >
      <h2 className="text-lg font-semibold tracking-tight">Add user</h2>
      <p className="mt-1 text-sm text-muted">
        The password you set here is what the new user signs in with. Share it
        with them directly — it cannot be read back later.
      </p>

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        <Field label="Name" value={name} onChange={setName} required />
        <Field
          label="Email"
          value={email}
          onChange={setEmail}
          type="email"
          required
        />
        <Field
          label="Password"
          value={password}
          onChange={setPassword}
          type="password"
          required
          hint="At least 8 characters"
        />
        <div>
          <label className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            Role
          </label>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value)}
            className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          >
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <Field
          label="Specialization"
          value={specialization}
          onChange={setSpecialization}
        />
        <Field label="License number" value={license} onChange={setLicense} />
      </div>

      <div className="mt-6 flex gap-2">
        <Button type="submit" loading={busy}>
          Create user
        </Button>
        <Button type="button" variant="ghost" onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
    </form>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
  required = false,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  required?: boolean;
  hint?: string;
}) {
  return (
    <div>
      <label className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
        {label}
      </label>
      <input
        type={type}
        value={value}
        required={required}
        onChange={(e) => onChange(e.target.value)}
        placeholder={hint}
        className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
      />
    </div>
  );
}

// ─── Reset password ──────────────────────────────────────────────
function ResetPasswordDialog({
  user,
  onClose,
  onError,
}: {
  user: DoctorPublicDTO;
  onClose: () => void;
  onError: (msg: string | null) => void;
}) {
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  /** Fills a random password and leaves it visible so the admin can
   *  copy it. This is the only moment it is ever readable — the server
   *  stores a bcrypt hash, which is one-way. */
  function generate() {
    const alphabet =
      "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
    const bytes = new Uint32Array(12);
    crypto.getRandomValues(bytes);
    setPw(Array.from(bytes, (b) => alphabet[b % alphabet.length]).join(""));
    setDone(false);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    onError(null);
    try {
      await adminResetPassword(user.id, pw);
      setDone(true);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not reset password");
      onClose();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <form
        onSubmit={submit}
        className="w-full max-w-md rounded-card border border-border bg-surface p-6"
      >
        <h2 className="text-lg font-semibold tracking-tight">Reset password</h2>
        <p className="mt-1 text-sm text-muted">
          Set a new password for <span className="text-foreground">{user.name}</span>{" "}
          ({user.email}).
        </p>

        <div className="mt-5">
          <label className="block text-xs font-semibold uppercase tracking-[0.12em] text-subtle">
            New password
          </label>
          <div className="mt-2 flex gap-2">
            <input
              type="text"
              value={pw}
              required
              minLength={8}
              maxLength={128}
              onChange={(e) => {
                setPw(e.target.value);
                setDone(false);
              }}
              placeholder="At least 8 characters"
              className="flex-1 rounded-md border border-border bg-background px-3 py-2 font-mono text-sm"
            />
            <Button type="button" variant="secondary" onClick={generate}>
              Generate
            </Button>
          </div>
          <p className="mt-2 text-xs text-muted">
            Shown in plain text so you can copy it now — it cannot be retrieved
            afterwards.
          </p>
        </div>

        {done ? (
          <div className="mt-5 rounded-md border border-emerald-500/40 bg-emerald-500/5 p-3 text-sm text-emerald-500">
            Password updated. Copy it before closing.
          </div>
        ) : null}

        <div className="mt-6 flex gap-2">
          <Button type="submit" loading={busy} disabled={done}>
            {done ? "Saved" : "Set password"}
          </Button>
          <Button type="button" variant="ghost" onClick={onClose}>
            {done ? "Done" : "Cancel"}
          </Button>
        </div>
      </form>
    </div>
  );
}
