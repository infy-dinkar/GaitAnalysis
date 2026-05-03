import { Activity, FileText, Zap } from "lucide-react";
import { Section } from "@/components/ui/Section";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

const FEATURES = [
  {
    icon: Activity,
    title: "Markerless Tracking",
    body:
      "33-point body model from any standard camera. No suits, no reflective markers, no specialised hardware — just video.",
  },
  {
    icon: Zap,
    title: "Real-Time Analysis",
    body:
      "Pose estimation runs entirely in the browser at 30+ FPS. Live angle feedback, recording windows, instant peak metrics.",
  },
  {
    icon: FileText,
    title: "Clinical Reports",
    body:
      "Joint angle charts, gait cycle breakdown, normal-range overlays. Export PDF reports your patients and referrers can read.",
  },
];

export function Features() {
  return (
    <Section id="product">
      <div className="max-w-2xl">
        <Badge>Capabilities</Badge>
        <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">
          Built for movement<br />professionals.
        </h2>
        <p className="mt-5 text-lg text-muted">
          Every measurement you would expect from a motion-capture lab — without the lab.
        </p>
      </div>

      <div className="mt-16 grid gap-6 md:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, body }) => (
          <Card key={title} variant="interactive" className="flex h-full flex-col p-8">
            <span className="flex h-12 w-12 items-center justify-center rounded-lg bg-accent/10 text-accent">
              <Icon className="h-6 w-6" />
            </span>
            <h3 className="mt-6 text-xl font-semibold tracking-tight">{title}</h3>
            <p className="mt-3 text-sm leading-relaxed text-muted">{body}</p>
          </Card>
        ))}
      </div>
    </Section>
  );
}
