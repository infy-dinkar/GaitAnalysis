import { Building2, GraduationCap, Stethoscope } from "lucide-react";
import { Section } from "@/components/ui/Section";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

const CASES = [
  {
    icon: Stethoscope,
    title: "Clinics",
    body:
      "Outpatient physio and rehab. Track ROM progression visit-over-visit with objective angle data, not subjective notes.",
  },
  {
    icon: Building2,
    title: "Hospitals",
    body:
      "Pre-op screening and post-op recovery monitoring. Standardised gait reports across departments and clinicians.",
  },
  {
    icon: GraduationCap,
    title: "Universities",
    body:
      "Undergraduate kinesiology labs and movement-science research. Reproducible measurements without a motion capture rig.",
  },
];

export function UseCases() {
  return (
    <Section id="use-cases">
      <div className="max-w-2xl">
        <Badge>Who uses it</Badge>
        <h2 className="mt-5 text-3xl font-semibold tracking-tight md:text-5xl">
          From outpatient room<br />to research lab.
        </h2>
      </div>

      <div className="mt-16 grid gap-6 md:grid-cols-3">
        {CASES.map(({ icon: Icon, title, body }) => (
          <Card key={title} className="flex h-full flex-col p-8">
            <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-accent/10 text-accent">
              <Icon className="h-5 w-5" />
            </span>
            <h3 className="mt-6 text-lg font-semibold tracking-tight">{title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
          </Card>
        ))}
      </div>
    </Section>
  );
}
