# Remediation guide

## DRC-COMP-002 Missing close value on won deals
Back-fill from the billing or contract system. If unavailable, flag the deal for the sales owner to confirm and exclude it from revenue reports until resolved. Add a required-field rule at data entry.

## DRC-INTG-001 Orphan account references
Check spelling and casing against the accounts table first. Create missing accounts only after confirming they are real customers. Replace free text with a lookup field.

## DRC-CONS-001 Invalid date sequences
Confirm the true dates with the deal owner. Swap only when clearly entered in the wrong order. Add an entry-time check that close_date is not before engage_date.

## DRC-VALD-002 Inconsistent stage names
Standardise to the four allowed stages with a mapping table, then lock the field to a picklist.
