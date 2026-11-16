"""Version-pinned offline card metadata. No game input or credential access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import time

if __package__:
    from . import cli
else:
    import cli


class Catalog:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        self.meta = dict(self.db.execute('SELECT key,value FROM metadata'))

    def get(self, identifier):
        column = 'dbf_id' if isinstance(identifier, int) else 'card_id'
        row = self.db.execute(f'SELECT data FROM cards WHERE {column}=?', (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def search(self, name):
        return [json.loads(row[0]) for row in self.db.execute(
            'SELECT data FROM cards WHERE name_zh LIKE ? OR name_en LIKE ? ORDER BY card_id LIMIT 100',
            ('%'+name+'%', '%'+name+'%'))]

    def close(self):
        self.db.close()


def build(zh, en, output, build_id):
    if output.exists():
        raise FileExistsError(output)
    chinese = json.loads(zh.read_text()); english = {c['id']: c for c in json.loads(en.read_text())}
    if len({c['id'] for c in chinese}) != len(chinese):
        raise ValueError('duplicate card ids')
    if {c['id'] for c in chinese} != set(english):
        raise ValueError('locale card id sets differ')
    structural = ('type','cost','attack','health','armor','techLevel','heroPowerDbfId',
                  'isBattlegroundsPoolMinion','isBattlegroundsPoolSpell','battlegroundsHero','mechanics','playRequirements')
    for card in chinese:
        if any(card.get(key) != english[card['id']].get(key) for key in structural):
            raise ValueError('locale rule fields differ for '+card['id'])
    output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output)
    with db:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('CREATE TABLE cards(card_id TEXT PRIMARY KEY,dbf_id INTEGER UNIQUE,name_zh TEXT,name_en TEXT,data TEXT NOT NULL)')
        db.execute('CREATE INDEX names_zh ON cards(name_zh)')
        for key, value in {'build': str(build_id), 'zh_sha256': cli.sha(zh), 'en_sha256': cli.sha(en),
                           'count': str(len(chinese)), 'created_utc': cli.now()}.items():
            db.execute('INSERT INTO metadata VALUES (?,?)', (key, value))
        for card in chinese:
            data = {**card, 'name_en': english[card['id']].get('name'), 'text_en': english[card['id']].get('text')}
            db.execute('INSERT INTO cards VALUES (?,?,?,?,?)', (card['id'], card['dbfId'], card.get('name'),
                       data['name_en'], json.dumps(data, ensure_ascii=False)))
    db.close()
    return {'build': str(build_id), 'cards': len(chinese), 'database': str(output), 'sha256': cli.sha(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    import_command = commands.add_parser('build')
    for name in ('zh', 'en', 'output'): import_command.add_argument('--'+name, type=Path, required=True)
    import_command.add_argument('--build', required=True)
    lookup = commands.add_parser('lookup'); lookup.add_argument('--db', type=Path, required=True)
    query = lookup.add_mutually_exclusive_group(required=True)
    query.add_argument('--id'); query.add_argument('--name')
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.zh, args.en, args.output, args.build)
    else:
        catalog = Catalog(args.db)
        result = catalog.search(args.name) if args.name else catalog.get(int(args.id) if args.id.isdigit() else args.id)
        if isinstance(result, dict) and result.get('heroPowerDbfId'):
            result = {**result, 'hero_power': catalog.get(result['heroPowerDbfId'])}
        catalog.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
