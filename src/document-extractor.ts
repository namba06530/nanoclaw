import { createRequire } from 'module';

import mammoth from 'mammoth';

const require = createRequire(import.meta.url);
// pdf-parse has no ESM default export — use require
const pdfParse = require('pdf-parse') as (buf: Buffer) => Promise<{ text: string }>;

const MAX_CHARS = 20_000;

export async function extractTextFromBuffer(
  buffer: Buffer,
  mimeType: string,
  filename: string,
): Promise<string> {
  try {
    let text = '';

    if (mimeType === 'application/pdf' || filename.endsWith('.pdf')) {
      const data = await pdfParse(buffer);
      text = data.text;
    } else if (
      mimeType ===
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
      filename.endsWith('.docx')
    ) {
      const result = await mammoth.extractRawText({ buffer });
      text = result.value;
    } else if (
      mimeType.startsWith('text/') ||
      ['.txt', '.md', '.csv', '.json'].some((ext) => filename.endsWith(ext))
    ) {
      text = buffer.toString('utf-8');
    } else {
      return `[Format non supporté pour l'extraction : ${filename}]`;
    }

    text = text.trim();
    if (text.length > MAX_CHARS) {
      return (
        text.slice(0, MAX_CHARS) +
        `\n\n[... contenu tronqué à ${MAX_CHARS} caractères]`
      );
    }
    return text || '[Fichier vide ou non lisible]';
  } catch (err: any) {
    return `[Erreur lors de l'extraction : ${err.message}]`;
  }
}
